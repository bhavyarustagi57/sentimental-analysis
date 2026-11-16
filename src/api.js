// Shared API helper. In dev, Vite proxies /api -> http://127.0.0.1:8000 (see vite.config.js).
export const API_BASE = import.meta.env.VITE_API_BASE_URL || '/api'

export async function request(path, options) {
  let response
  try {
    response = await fetch(`${API_BASE}${path}`, options)
  } catch {
    throw new Error('Cannot reach the backend. Is it running on port 8000?')
  }
  const body = await response.json().catch(() => ({}))
  if (!response.ok) {
    const detail = typeof body.detail === 'string' ? body.detail : Array.isArray(body.detail) ? body.detail.map((item) => item.msg).join('; ') : ''
    throw new Error(detail || `Request failed (${response.status})`)
  }
  return body
}
