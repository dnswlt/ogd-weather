package main

import (
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
)

func main() {
	// Static file handler
	fs := http.FileServer(http.Dir("static"))
	http.Handle("/", fs)

	// Reverse proxy to FastAPI for /stations/*
	target, _ := url.Parse("http://127.0.0.1:8000")
	proxy := httputil.NewSingleHostReverseProxy(target)

	http.Handle("/stations/", proxy)

	log.Println("Go API server on http://localhost:8080")
	log.Fatal(http.ListenAndServe(":8080", nil))
}
