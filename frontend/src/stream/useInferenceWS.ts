import { useCallback, useEffect, useRef, useState } from 'react';
import type { ServerMessage } from './protocol';

type Status = 'connecting' | 'open' | 'closed' | 'error';

interface UseInferenceWSResult {
  status: Status;
  send: (data: ArrayBuffer) => boolean;
  inFlight: React.RefObject<boolean>;
  lastMessage: ServerMessage | null;
}

export function useInferenceWS(url: string): UseInferenceWSResult {
  const wsRef = useRef<WebSocket | null>(null);
  const inFlight = useRef<boolean>(false);
  const [status, setStatus] = useState<Status>('connecting');
  const [lastMessage, setLastMessage] = useState<ServerMessage | null>(null);

  useEffect(() => {
    const ws = new WebSocket(url);
    ws.binaryType = 'arraybuffer';
    wsRef.current = ws;

    ws.onopen = () => setStatus('open');
    ws.onclose = () => setStatus('closed');
    ws.onerror = () => setStatus('error');
    ws.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data as string) as ServerMessage;
        inFlight.current = false;
        setLastMessage(msg);
      } catch {
        // ignore malformed
      }
    };

    return () => {
      ws.close();
      wsRef.current = null;
    };
  }, [url]);

  const send = useCallback((data: ArrayBuffer): boolean => {
    const ws = wsRef.current;
    if (ws?.readyState !== WebSocket.OPEN) return false;
    if (inFlight.current) return false; // backpressure: drop frame
    inFlight.current = true;
    ws.send(data);
    return true;
  }, []);

  return { status, send, inFlight, lastMessage };
}
