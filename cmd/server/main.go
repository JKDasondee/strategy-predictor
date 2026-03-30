package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

func main() {
	port := "8080"
	if p := os.Getenv("PORT"); p != "" {
		port = p
	}

	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", handleHealth)
	mux.HandleFunc("GET /analyze", handleAnalyze)

	srv := &http.Server{
		Addr:         ":" + port,
		Handler:      mux,
		ReadTimeout:  10 * time.Second,
		WriteTimeout: 30 * time.Second,
	}

	log.Printf("listening on :%s", port)
	log.Fatal(srv.ListenAndServe())
}

func handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{
		"status": "ok",
		"ts":     time.Now().UTC().Format(time.RFC3339),
	})
}

func handleAnalyze(w http.ResponseWriter, r *http.Request) {
	assets := r.URL.Query().Get("assets")
	chain := r.URL.Query().Get("chain")

	if assets == "" {
		httpErr(w, http.StatusBadRequest, "missing assets param")
		return
	}
	if chain == "" {
		chain = "8453"
	}

	if _, err := strconv.Atoi(chain); err != nil {
		httpErr(w, http.StatusBadRequest, "invalid chain id")
		return
	}

	pairs := strings.Split(assets, ",")
	for _, p := range pairs {
		parts := strings.SplitN(strings.TrimSpace(p), ":", 2)
		if len(parts) != 2 {
			httpErr(w, http.StatusBadRequest, fmt.Sprintf("invalid asset pair: %s", p))
			return
		}
		if _, err := strconv.ParseFloat(parts[1], 64); err != nil {
			httpErr(w, http.StatusBadRequest, fmt.Sprintf("invalid weight: %s", parts[1]))
			return
		}
	}

	cmd := exec.Command("python", "-m", "src.predict",
		"--assets", assets,
		"--chain", chain,
	)
	cmd.Dir = projectRoot()

	out, err := cmd.Output()
	if err != nil {
		if ee, ok := err.(*exec.ExitError); ok {
			httpErr(w, http.StatusInternalServerError, string(ee.Stderr))
		} else {
			httpErr(w, http.StatusInternalServerError, err.Error())
		}
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.Write(out)
}

func httpErr(w http.ResponseWriter, code int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(map[string]string{"error": msg})
}

func projectRoot() string {
	if d := os.Getenv("PROJECT_ROOT"); d != "" {
		return d
	}
	return "."
}
