// Prefer relative paths by default (use Vite proxy in dev). If VITE_BACKEND_BASE is
// explicitly provided, use it (production or custom deployments).
const __base = import.meta.env.VITE_BACKEND_BASE as string | undefined;
export const backendBase = (__base && __base.length > 0) ? __base : "";

export async function apiPost(path: string, body: any) {
  const res = await fetch(`${backendBase}${path}`, {
    method: "POST",
    credentials: "include",
    headers: { 
      "Content-Type": "application/json",
      "Accept": "application/json"
    },
    body: JSON.stringify(body),
  });
  
  if (!res.ok) {
    let errorMessage = res.statusText;
    try {
      const json = await res.json();
      errorMessage = json?.detail || json?.message || JSON.stringify(json);
    } catch {
      // If JSON parsing fails, use status text
    }
    throw new Error(errorMessage);
  }
  
  try {
    const json = await res.json();
    return json;
  } catch {
    return null;
  }
}

export async function apiGet<T = unknown>(path: string): Promise<T> {
  const res = await fetch(`${backendBase}${path}`, {
    method: "GET",
    credentials: "include",
    headers: { "Accept": "application/json" }
  });

  if (!res.ok) {
    let errorMessage = res.statusText;
    try {
      const json = await res.json();
      errorMessage = json?.detail || json?.message || JSON.stringify(json);
    } catch {}
    throw new Error(errorMessage);
  }

  try {
    return (await res.json()) as T;
  } catch {
    // @ts-expect-error allow null for endpoints with no body
    return null;
  }
}

export async function apiDelete<T = unknown>(path: string): Promise<T> {
  const res = await fetch(`${backendBase}${path}`, {
    method: "DELETE",
    credentials: "include",
    headers: { "Accept": "application/json" }
  });

  if (!res.ok) {
    let errorMessage = res.statusText;
    try {
      const json = await res.json();
      errorMessage = json?.detail || json?.message || JSON.stringify(json);
    } catch {}
    throw new Error(errorMessage);
  }

  try {
    return (await res.json()) as T;
  } catch {
    // Some DELETEs may not return a body
    // @ts-expect-error allow null
    return null;
  }
}
