// SSE consumer for POST /ask. Native EventSource only supports GET, so we
// roll our own parser on top of fetch + ReadableStream. Handles the three
// frame types the backend emits: `sources` (JSON), `token` (text), `done`.

import { API_BASE } from "./client";
import type { AskSource, SearchFilterPayload } from "../types";

export interface AskRequest {
  question: string;
  top_k?: number;
  rerank?: boolean;
  filter?: SearchFilterPayload;
  max_tokens?: number;
  temperature?: number;
}

export interface AskHandlers {
  onSources: (sources: AskSource[]) => void;
  onToken: (token: string) => void;
  onDone: () => void;
  onError: (err: unknown) => void;
  signal?: AbortSignal;
}

export async function askStream(req: AskRequest, h: AskHandlers): Promise<void> {
  try {
    const resp = await fetch(`${API_BASE}/ask`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "text/event-stream",
      },
      body: JSON.stringify(req),
      signal: h.signal,
    });

    if (!resp.ok) {
      let detail = `HTTP ${resp.status}`;
      try {
        const body = await resp.json();
        if (body?.detail) detail = `${detail}: ${JSON.stringify(body.detail)}`;
      } catch {
        /* ignore */
      }
      throw new Error(detail);
    }
    if (!resp.body) throw new Error("Streaming response has no body");

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // SSE events are separated by a blank line.
      let idx;
      while ((idx = buffer.indexOf("\n\n")) >= 0) {
        const raw = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);
        dispatchEvent(raw, h);
      }
    }
    if (buffer.trim()) dispatchEvent(buffer, h);
    h.onDone();
  } catch (e) {
    if ((e as Error).name === "AbortError") {
      h.onDone();
      return;
    }
    h.onError(e);
  }
}

function dispatchEvent(raw: string, h: AskHandlers): void {
  let event = "message";
  const dataLines: string[] = [];
  for (const line of raw.split("\n")) {
    if (line.startsWith(":")) continue; // SSE comment
    if (line.startsWith("event:")) {
      event = line.slice(6).trimStart();
    } else if (line.startsWith("data:")) {
      // SSE: strip a single leading space if present (per spec)
      let v = line.slice(5);
      if (v.startsWith(" ")) v = v.slice(1);
      dataLines.push(v);
    }
  }
  const data = dataLines.join("\n");
  if (event === "sources") {
    try {
      h.onSources(JSON.parse(data) as AskSource[]);
    } catch {
      /* ignore parse errors on sources frame */
    }
  } else if (event === "token") {
    h.onToken(data);
  } else if (event === "done") {
    h.onDone();
  }
}
