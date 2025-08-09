package main

import (
	"flag"
	"log"
	"os"
	"strconv"

	"github.com/dnswlt/ogd-weather/service/api"
)

func main() {
	// Get defaults from env, if set.
	chartServiceEndpoint := os.Getenv("OGD_CHART_SERVICE_ENDPOINT")
	if chartServiceEndpoint == "" {
		chartServiceEndpoint = "http://127.0.0.1:8000" // local default
	}
	addr := os.Getenv("OGD_API_ADDR")
	if addr == "" {
		addr = "127.0.0.1:8080" // local default
	}
	cacheSize, err := strconv.Atoi(os.Getenv("OGD_CACHE_SIZE"))
	if err != nil {
		cacheSize = 0
	}

	var opts api.ServerOptions
	flag.StringVar(&opts.Addr, "addr", addr, "Address that the server listens on.")
	flag.StringVar(&opts.ChartServiceEndpoint, "chart-service-endpoint", chartServiceEndpoint, "URL of the chart service backend.")
	flag.StringVar(&opts.BaseDir, "base-dir", ".", "Base directory containing templates/ and static/ subdirectories.")
	flag.BoolVar(&opts.DebugMode, "debug", false, "If specified, the server runs in debug mode.")
	flag.BoolVar(&opts.LogRequests, "log-requests", false, "If true, the server write request logs.")
	flag.IntVar(&opts.CacheSize, "cache-size", cacheSize, "Approximate size of the response cache in bytes. (0 = disabled, -1 = unlimited)")
	useViteManifest := flag.Bool("vite-manifest", true, "Use Vite manifest.json for hashed filename mapping")
	flag.Parse()

	var moreOpts []api.ServerOption

	if token := os.Getenv("OGD_BEARER_TOKEN"); token != "" {
		moreOpts = append(moreOpts, api.WithBearerToken(token))
	}
	if *useViteManifest {
		moreOpts = append(moreOpts, api.WithViteManifest())
	}
	server, err := api.NewServer(opts, moreOpts...)
	if err != nil {
		log.Fatalf("Error creating server: %v", err)

	}
	log.Fatal(server.Serve())
}
