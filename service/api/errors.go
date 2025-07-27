package api

import (
	"fmt"
	"net/http"
)

// BackendError indicates that a call to the charts backend has failed.
type BackendError struct {
	backendURL string // URL that returned the error
	statusCode int    // HTTP status code
}

func (e *BackendError) BackendURL() string { return e.backendURL }
func (e *BackendError) StatusCode() int    { return e.statusCode }

func (e *BackendError) Error() string {
	return fmt.Sprintf("%s: %d %s", e.backendURL, e.statusCode, http.StatusText(e.statusCode))
}

// Unwrap allows errors.Unwrap / errors.Is / errors.As to work as expected.
func (e *BackendError) Unwrap() error {
	return nil
}

// Helper to create a new HTTPError without wrapping
func NewBackendError(backendURL string, code int) error {
	return &BackendError{
		backendURL: backendURL,
		statusCode: code,
	}
}
