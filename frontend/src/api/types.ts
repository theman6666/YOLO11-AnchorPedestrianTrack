// API response types from Flask backend

export interface DetectionResponse {
  ok: boolean
  count?: number
  image_url?: string
  video_url?: string
  stats?: VideoStats
  message: string
}

export interface VideoStats {
  frames: number
  avg_fps: number
}

export interface ApiError {
  message: string
  status?: number
  code?: string
}

export interface VideoFeedParams {
  camera_id: number
}
