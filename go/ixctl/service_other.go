//go:build !windows

package main

// hostService runs the host on the console: only Windows has a service manager here.
func hostService(_ string, o hostOptions) (uint32, error) { return consoleHost(o), nil }
