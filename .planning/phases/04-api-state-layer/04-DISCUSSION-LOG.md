# Phase 4: API & State Layer - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-03
**Phase:** 04-api-state-layer
**Mode:** Interactive discussion

---

## Camera Streaming

### Stream Source

| Option | Description | Selected |
|--------|-------------|----------|
| MJPEG stream | Use `/video_feed` endpoint streaming response | ✓ |
| Alternative protocol | HLS or other streaming protocols | |

**User's choice:** MJPEG stream

### Stream Format

| Option | Description | Selected |
|--------|-------------|----------|
| Direct URL | Set img src to `/video_feed?camera_id={id}&t={timestamp}` | ✓ |
| Fetch blob | Fetch and convert to blob URL | |

**User's choice:** Direct URL

---

## Error Handling

### Error Display

| Option | Description | Selected |
|--------|-------------|----------|
| Hybrid approach | Inline errors + StatusMonitor for critical errors | ✓ |
| Status bar only | All errors in StatusMonitor | |
| Panel focused | Detailed errors in panels, simple in StatusMonitor | |

**User's choice:** Hybrid approach

### Retry Logic

| Option | Description | Selected |
|--------|-------------|----------|
| No auto-retry | Fail immediately, show error, user manually retries | ✓ |
| Single retry | Auto-retry once on network error | |
| Axios interceptors | Handle retry in interceptors | |

**User's choice:** No auto-retry

---

## State Updates

### Handler Pattern

| Option | Description | Selected |
|--------|-------------|----------|
| async/await | Standard async/await with try-catch | ✓ |
| Promise chain | .then().catch() chaining | |

**User's choice:** async/await

### Update Timing

| Option | Description | Selected |
|--------|-------------|----------|
| Standard pattern | Set loading state before API, update results after response | ✓ |
| Reactive pattern | Use Vue watch/computed | |

**User's choice:** Standard pattern

---

## Loading States

### State Scope

| Option | Description | Selected |
|--------|-------------|----------|
| Use existing | Use processing.camera/image/video object already defined | ✓ |
| New pattern | Create new state management (composable/store) | |

**User's choice:** Use existing

---

## Claude's Discretion

Areas where user delegated decisions:
- Timeout values (60s for video already configured)
- Error message wording
- Status bar timing

---

## Deferred Ideas

- Auto-retry logic: User prefers manual retry over automatic retry
- Request cancellation: Not implementing abort controller
- Progress indicators: No upload progress bars
- Connection pooling: Single Axios instance sufficient

---

*Phase: 04-api-state-layer*
*Discussion logged: 2026-04-03*
