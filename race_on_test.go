//go:build race

package tokcount

// The race detector slows code down severalfold.
const raceSlowdown = 10
