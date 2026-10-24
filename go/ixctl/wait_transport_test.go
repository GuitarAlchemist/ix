package main

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"testing"
)

func TestWaitTimeoutAndRefusalsStayOnTheirTransport(t *testing.T) {
	in := newTestInstance(t, fakeSpec(t, "local", nil,
		fakeSvc{name: "pod", restart: restartNever}), "")
	c := dial(t, in)
	applied := applyWS(t, c, running("pod"))
	launch := waitPlane(t, in.plane, Request{Op: "wait", Wait: waitLaunch, Epoch: applied.Epoch,
		Generation: applied.Generation, Service: "pod", TimeoutMS: testWaitTimeoutMS})
	if launch.Run == nil {
		t.Fatalf("launch wait %+v", launch)
	}

	code, _, raw := httpDo(t, in, "POST", "/v1/wait",
		map[string]any{"wait": waitRun, "run": launch.Run, "timeout_ms": 20}, nil)
	var timeout WaitReply
	if err := json.Unmarshal(raw, &timeout); err != nil || code != 200 || timeout.Outcome != waitTimedOut {
		t.Fatalf("timeout: %d %s (%v)", code, raw, err)
	}
	stopped := applyWS(t, c, desiredOf(desiredStopped, "pod"))
	for _, body := range []map[string]any{
		{"wait": waitRun, "run": launch.Run},
		{"wait": waitLaunch, "epoch": stopped.Epoch, "generation": stopped.Generation,
			"service": "pod", "timeout_ms": testWaitTimeoutMS},
		{"wait": waitLaunch, "epoch": applied.Epoch, "generation": applied.Generation,
			"service": "missing", "timeout_ms": testWaitTimeoutMS},
		{"wait": waitRun, "run": launch.Run, "timeout_ms": waitTimeoutMax.Milliseconds() + 1},
	} {
		if code, _, raw := httpDo(t, in, "POST", "/v1/wait", body, nil); code != 400 {
			t.Fatalf("malformed wait accepted: %d %s", code, raw)
		}
	}
	if _, err := c.Do(context.Background(), Request{Op: "wait", Wait: waitRun, Run: launch.Run}); err == nil {
		t.Fatal("WebSocket wait without timeout_ms accepted")
	}
	if _, err := c.Do(context.Background(), Request{Op: "wait", Wait: waitLaunch, Epoch: applied.Epoch,
		Generation: applied.Generation, Service: "missing", TimeoutMS: testWaitTimeoutMS}); err == nil {
		t.Fatal("WebSocket wait for unknown service accepted")
	}
}

func TestWaitersAreBoundedAndDisconnectsReleaseTheirSlots(t *testing.T) {
	in := newTestInstance(t, fakeSpec(t, "local", nil,
		fakeSvc{name: "pod", restart: restartNever}), "")
	c := dial(t, in)
	applied := applyWS(t, c, running("pod"))
	launch := waitPlane(t, in.plane, Request{Op: "wait", Wait: waitLaunch, Epoch: applied.Epoch,
		Generation: applied.Generation, Service: "pod", TimeoutMS: testWaitTimeoutMS})
	if launch.Run == nil {
		t.Fatalf("launch wait %+v", launch)
	}
	status := in.plane.Status()
	if status.Limits.WaitTimeoutMS != waitTimeoutMax.Milliseconds() || status.Limits.Waiters != maxWaiters {
		t.Fatalf("wait limits %+v", status.Limits)
	}

	in.plane.waiters = make(chan struct{}, 1)
	in.plane.waiters <- struct{}{}
	body := map[string]any{"wait": waitRun, "run": launch.Run, "timeout_ms": testWaitTimeoutMS}
	if code, _, raw := httpDo(t, in, "POST", "/v1/wait", body, nil); code != 429 {
		t.Fatalf("full HTTP waiter set: %d %s", code, raw)
	}
	if _, err := c.Do(context.Background(), Request{Op: "wait", Wait: waitRun, Run: launch.Run,
		TimeoutMS: testWaitTimeoutMS}); err == nil || !strings.Contains(err.Error(), "too many waiters") {
		t.Fatalf("full WebSocket waiter set: %v", err)
	}
	<-in.plane.waiters

	httpCtx, cancelHTTP := context.WithCancel(context.Background())
	requestBody, _ := json.Marshal(body)
	request, _ := http.NewRequestWithContext(httpCtx, http.MethodPost,
		"http://"+in.addr+"/v1/wait", strings.NewReader(string(requestBody)))
	request.Header.Set("Authorization", "Bearer "+in.token)
	httpDone := make(chan error, 1)
	go func() {
		response, err := loopbackHTTP.Do(request)
		if response != nil {
			response.Body.Close()
		}
		httpDone <- err
	}()
	waitForWaiterCount(t, in.plane, 1)
	cancelHTTP()
	<-httpDone
	waitForWaiterCount(t, in.plane, 0)

	ws := dial(t, in)
	wsDone := make(chan error, 1)
	go func() {
		_, err := ws.Do(context.Background(), Request{Op: "wait", Wait: waitRun, Run: launch.Run,
			TimeoutMS: testWaitTimeoutMS})
		wsDone <- err
	}()
	waitForWaiterCount(t, in.plane, 1)
	ws.Close()
	if err := <-wsDone; err == nil {
		t.Fatal("closed WebSocket wait returned no transport error")
	}
	waitForWaiterCount(t, in.plane, 0)
}

func TestShutdownEndsAnInFlightWaitWithAnError(t *testing.T) {
	in := newTestInstance(t, fakeSpec(t, "local", nil,
		fakeSvc{name: "pod", restart: restartNever}), "")
	c := dial(t, in)
	applied := applyWS(t, c, running("pod"))
	launch := waitPlane(t, in.plane, Request{Op: "wait", Wait: waitLaunch, Epoch: applied.Epoch,
		Generation: applied.Generation, Service: "pod", TimeoutMS: testWaitTimeoutMS})
	if launch.Run == nil {
		t.Fatalf("launch wait %+v", launch)
	}

	done := make(chan error, 1)
	go func() {
		_, err := in.plane.Request(context.Background(), Request{
			Op: "wait", Wait: waitRun, Run: launch.Run, TimeoutMS: waitTimeoutMax.Milliseconds()})
		done <- err
	}()
	waitForWaiterCount(t, in.plane, 1)
	in.plane.RequestShutdown()
	if err := <-done; err == nil || statusOf(err) != 503 || !strings.Contains(err.Error(), "shutting down") {
		t.Fatalf("shutdown wait error %v", err)
	}
	waitForWaiterCount(t, in.plane, 0)
}
