//go:build !windows

package main

// hostService runs the host on the console: only Windows has a service manager here.
func hostService(_ string, setup func() (hostOptions, func(), error)) (uint32, error) {
	o, done, err := setup()
	if err != nil {
		return 0, err
	}
	defer done()
	return consoleHost(o), nil
}
