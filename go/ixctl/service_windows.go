//go:build windows

package main

import (
	"golang.org/x/sys/windows/svc"
)

// hostService runs the host under the SCM when the SCM started this process, else on the
// console. setup (spec, state directory, log) runs after the SCM is connected.
func hostService(name string, setup func() (hostOptions, func(), error)) (uint32, error) {
	isService, err := svc.IsWindowsService()
	if err != nil {
		return 0, err
	}
	if !isService {
		o, done, err := setup()
		if err != nil {
			return 0, err
		}
		defer done()
		return consoleHost(o), nil
	}
	var code uint32
	h := &scmHandler{setup: setup, run: func(o hostOptions, c <-chan hostControl, r func(hostStatus)) uint32 {
		code = runHost(o, c, r)
		return code
	}}
	if err := svc.Run(name, h); err != nil {
		return 0, err
	}
	return code, nil
}

// scmHandler carries the SCM's requests to a host and its reports back. The final stopped
// report is not sent: the svc package reports it from Execute's return.
type scmHandler struct {
	setup func() (hostOptions, func(), error)
	run   func(hostOptions, <-chan hostControl, func(hostStatus)) uint32
}

func (h *scmHandler) Execute(_ []string, r <-chan svc.ChangeRequest, s chan<- svc.Status) (bool, uint32) {
	s <- scmStatus(hostStatus{State: hostStartPending, Checkpoint: 1, WaitHint: hostStartHint})
	o, done, err := h.setup()
	if err != nil {
		return true, exitStartFailed
	}
	defer done()
	controls := make(chan hostControl)
	stop := make(chan struct{})
	go func() {
		for {
			var c svc.ChangeRequest
			var ok bool
			select {
			case c, ok = <-r:
				if !ok {
					return
				}
			case <-stop:
				return
			}
			hc := hostControl{Source: "scm"}
			switch c.Cmd {
			case svc.Stop:
				hc.Cmd = controlStop
			case svc.Shutdown:
				hc.Cmd = controlShutdown
			case svc.Interrogate:
				hc.Cmd = controlInterrogate
			default:
				continue
			}
			select {
			case controls <- hc:
			case <-stop:
				return
			}
		}
	}()
	code := h.run(o, controls, func(st hostStatus) {
		if st.State != hostStopped {
			s <- scmStatus(st)
		}
	})
	close(stop)
	return code != exitOK, code
}

func scmStatus(st hostStatus) svc.Status {
	out := svc.Status{CheckPoint: st.Checkpoint, WaitHint: uint32(st.WaitHint.Milliseconds())}
	switch st.State {
	case hostStartPending:
		out.State = svc.StartPending
	case hostRunning:
		out.State = svc.Running
	case hostStopPending:
		out.State = svc.StopPending
	default:
		out.State = svc.Stopped
	}
	if st.Accepts {
		out.Accepts = svc.AcceptStop | svc.AcceptShutdown
	}
	return out
}
