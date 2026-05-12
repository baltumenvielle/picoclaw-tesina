#!/usr/bin/env python3
"""
Benchmark script para PicoClaw en RPi4 4GB
Mide latencia, tokens/s, uso de CPU/RAM y tasa de éxito de tool calling.
"""

import subprocess
import time
import json
import os
import re
import csv
import datetime
import threading
import psutil

# --- Configuración ---
PICOCLAW_BIN = os.path.expanduser("~/picoclaw/picoclaw")
LLAMA_SERVER_URL = "http://127.0.0.1:8080"
RESULTS_FILE = os.path.expanduser("~/benchmark_results.csv")
LLAMA_LOG_FILE = "/tmp/llama_server.log"

PROMPTS = [
    {
        "id": 1,
        "prompt": "Create a reminder for tomorrow at 9am: meeting with Vanesa",
        "expected_tool": "write_file",
    },
    {
        "id": 2,
        "prompt": "Schedule a meeting with the team on Friday at 3pm",
        "expected_tool": "write_file",
    },
    {
        "id": 3,
        "prompt": "What reminders do I currently have pending?",
        "expected_tool": "read_file",
    },
    {
        "id": 4,
        "prompt": "Write down that I need to call Juan this afternoon",
        "expected_tool": "write_file",
    },
    {
        "id": 5,
        "prompt": "Do I have anything scheduled for this week?",
        "expected_tool": "read_file",
    },
]

SESSIONS_DIR = os.path.expanduser("~/.picoclaw/workspace/sessions")


def clear_picoclaw_sessions():
    """Borra el historial de sesiones de PicoClaw para evitar que el contexto crezca."""
    if not os.path.exists(SESSIONS_DIR):
        return
    deleted = 0
    for f in os.listdir(SESSIONS_DIR):
        if f.endswith(".jsonl") or f.endswith(".json"):
            os.remove(os.path.join(SESSIONS_DIR, f))
            deleted += 1
    if deleted:
        print(f"  [session] Borradas {deleted} sesiones previas")


def get_llama_server_pid():
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if "llama-server" in " ".join(proc.info["cmdline"] or []):
                return proc.info["pid"]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return None


def sample_resources(pid, samples, interval=0.5, stop_event=None):
    """Muestrea CPU y RAM del proceso llama-server en un thread separado."""
    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    while not (stop_event and stop_event.is_set()):
        try:
            cpu = proc.cpu_percent(interval=None)
            mem = proc.memory_info().rss / 1024 / 1024  # MB
            samples.append({"cpu": cpu, "mem_mb": mem})
        except psutil.NoSuchProcess:
            break
        time.sleep(interval)


def parse_llama_timing(log_text):
    """Extrae métricas de timing del log de llama-server."""
    result = {
        "n_tokens_prompt": None,
        "prompt_eval_ms": None,
        "prompt_tps": None,
        "eval_ms": None,
        "eval_tps": None,
        "total_ms": None,
        "n_tokens_generated": None,
    }

    # n_tokens del prompt
    m = re.search(r"task\.n_tokens = (\d+)", log_text)
    if m:
        result["n_tokens_prompt"] = int(m.group(1))

    # prompt eval time
    m = re.search(
        r"prompt eval time =\s+([\d.]+) ms /\s+(\d+) tokens.*?([\d.]+) tokens per second",
        log_text,
    )
    if m:
        result["prompt_eval_ms"] = float(m.group(1))
        result["prompt_tps"] = float(m.group(3))

    # eval time (generación)
    m = re.search(
        r"\s+eval time =\s+([\d.]+) ms /\s+(\d+) tokens.*?([\d.]+) tokens per second",
        log_text,
    )
    if m:
        result["eval_ms"] = float(m.group(1))
        result["n_tokens_generated"] = int(m.group(2))
        result["eval_tps"] = float(m.group(3))

    # total time
    m = re.search(r"total time =\s+([\d.]+) ms", log_text)
    if m:
        result["total_ms"] = float(m.group(1))

    return result


def detect_tool_call(picoclaw_output):
    """Detecta qué tool llamó el agente en el output de PicoClaw."""
    tools = ["write_file", "read_file", "list_dir"]
    for tool in tools:
        if tool in picoclaw_output:
            return tool
    return None


def run_benchmark_prompt(prompt_data, llama_pid, run_number):
    """Ejecuta un prompt y recolecta métricas."""
    print(f"\n[{run_number}] Prompt {prompt_data['id']}: {prompt_data['prompt'][:60]}...")

    # Limpiar sesión para que el contexto no crezca entre prompts
    clear_picoclaw_sessions()

    # Marca de tiempo del log antes del request
    log_start_marker = time.time()

    # Monitoreo de recursos en background
    samples = []
    stop_event = threading.Event()
    monitor_thread = threading.Thread(
        target=sample_resources, args=(llama_pid, samples, 0.5, stop_event)
    )
    monitor_thread.start()

    # Ejecutar PicoClaw
    wall_start = time.time()
    try:
        result = subprocess.run(
            [PICOCLAW_BIN, "agent", "-m", prompt_data["prompt"]],
            capture_output=True,
            text=True,
            timeout=700,
        )
        wall_end = time.time()
        picoclaw_output = result.stdout + result.stderr
        success = result.returncode == 0 or "🦞" in picoclaw_output
    except subprocess.TimeoutExpired:
        wall_end = time.time()
        picoclaw_output = "TIMEOUT"
        success = False

    stop_event.set()
    monitor_thread.join()

    wall_time_s = wall_end - wall_start

    # Leer log de llama-server
    llama_timing = {}
    try:
        with open(LLAMA_LOG_FILE) as f:
            log_content = f.read()
        # Solo las líneas después de log_start_marker (aproximado por contenido reciente)
        llama_timing = parse_llama_timing(log_content)
    except FileNotFoundError:
        print("  WARN: No se encontró el log de llama-server")

    # Detectar tool call
    tool_called = detect_tool_call(picoclaw_output)
    tool_correct = tool_called == prompt_data["expected_tool"]

    # Estadísticas de recursos
    cpu_avg = sum(s["cpu"] for s in samples) / len(samples) if samples else 0
    cpu_max = max((s["cpu"] for s in samples), default=0)
    mem_avg = sum(s["mem_mb"] for s in samples) / len(samples) if samples else 0
    mem_max = max((s["mem_mb"] for s in samples), default=0)

    metrics = {
        "timestamp": datetime.datetime.now().isoformat(),
        "run_number": run_number,
        "prompt_id": prompt_data["id"],
        "prompt": prompt_data["prompt"],
        "expected_tool": prompt_data["expected_tool"],
        "tool_called": tool_called,
        "tool_correct": tool_correct,
        "success": success,
        "wall_time_s": round(wall_time_s, 2),
        "n_tokens_prompt": llama_timing.get("n_tokens_prompt"),
        "prompt_eval_ms": llama_timing.get("prompt_eval_ms"),
        "prompt_tps": llama_timing.get("prompt_tps"),
        "n_tokens_generated": llama_timing.get("n_tokens_generated"),
        "eval_ms": llama_timing.get("eval_ms"),
        "eval_tps": llama_timing.get("eval_tps"),
        "total_llama_ms": llama_timing.get("total_ms"),
        "cpu_avg_pct": round(cpu_avg, 1),
        "cpu_max_pct": round(cpu_max, 1),
        "mem_avg_mb": round(mem_avg, 1),
        "mem_max_mb": round(mem_max, 1),
        "picoclaw_response": picoclaw_output.strip()[-200:],  # últimos 200 chars
    }

    print(f"  Wall time: {wall_time_s:.1f}s")
    print(f"  Tool llamada: {tool_called} (esperada: {prompt_data['expected_tool']}) → {'✓' if tool_correct else '✗'}")
    if llama_timing.get("prompt_tps"):
        print(f"  Prompt TPS: {llama_timing['prompt_tps']:.2f} | Gen TPS: {llama_timing.get('eval_tps', 'N/A')}")
    print(f"  CPU avg/max: {cpu_avg:.1f}% / {cpu_max:.1f}% | RAM avg/max: {mem_avg:.0f}MB / {mem_max:.0f}MB")

    return metrics


def write_results(results):
    if not results:
        return
    fieldnames = list(results[0].keys())
    file_exists = os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(results)
    print(f"\nResultados guardados en: {RESULTS_FILE}")


def print_summary(results):
    print("\n" + "=" * 60)
    print("RESUMEN DEL BENCHMARK")
    print("=" * 60)

    valid = [r for r in results if r["success"] and r["wall_time_s"] < 700]
    if not valid:
        print("No hay resultados válidos.")
        return

    wall_times = [r["wall_time_s"] for r in valid]
    tool_correct = sum(1 for r in valid if r["tool_correct"])
    prompt_tps = [r["prompt_tps"] for r in valid if r["prompt_tps"]]
    eval_tps = [r["eval_tps"] for r in valid if r["eval_tps"]]

    print(f"Prompts ejecutados: {len(results)}")
    print(f"Exitosos: {len(valid)}")
    print(f"Tool calling correcto: {tool_correct}/{len(valid)} ({100*tool_correct/len(valid):.0f}%)")
    print(f"\nLatencia wall time:")
    print(f"  Min: {min(wall_times):.1f}s | Max: {max(wall_times):.1f}s | Avg: {sum(wall_times)/len(wall_times):.1f}s")
    if prompt_tps:
        print(f"\nPrompt processing TPS: avg {sum(prompt_tps)/len(prompt_tps):.2f}")
    if eval_tps:
        print(f"Generación TPS: avg {sum(eval_tps)/len(eval_tps):.2f}")
    print("=" * 60)


def main():
    print("=== Benchmark PicoClaw + Qwen2.5 0.5B en RPi4 4GB ===")
    print(f"Resultados: {RESULTS_FILE}")
    print(f"LLM server: {LLAMA_SERVER_URL}")

    llama_pid = get_llama_server_pid()
    if not llama_pid:
        print("WARN: No se encontró el proceso llama-server. Las métricas de CPU/RAM no estarán disponibles.")
    else:
        print(f"llama-server PID: {llama_pid}")

    # Configurar llama-server para loguear a archivo
    print(f"\nNOTA: Para capturar métricas de tokens, el llama-server tiene que estar")
    print(f"corriendo con output redirigido a {LLAMA_LOG_FILE}")
    print(f"Ejemplo: llama-server ... > {LLAMA_LOG_FILE} 2>&1")
    print()

    all_results = []

    # Ronda 1: prompt frío (primera ejecución)
    print("\n--- RONDA 1: Prompt frío ---")
    for prompt_data in PROMPTS:
        metrics = run_benchmark_prompt(prompt_data, llama_pid, run_number=1)
        all_results.append(metrics)
        time.sleep(2)  # pausa entre prompts

    # Ronda 2: prompt caliente (cache activo)
    print("\n--- RONDA 2: Prompt con cache ---")
    for prompt_data in PROMPTS:
        metrics = run_benchmark_prompt(prompt_data, llama_pid, run_number=2)
        all_results.append(metrics)
        time.sleep(2)

    write_results(all_results)
    print_summary(all_results)


if __name__ == "__main__":
    main()
