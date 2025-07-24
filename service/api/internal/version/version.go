package version

// These are overridden at build time via -ldflags.
var (
	Version   = "dev"     // default when running locally
	GitCommit = "unknown" // short SHA
	BuildTime = "unknown" // RFC3339 UTC timestamp
)

func FullVersion() string {
	return Version + " (" + GitCommit + ", built " + BuildTime + ")"
}
