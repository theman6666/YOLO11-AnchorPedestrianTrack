import axios, { AxiosError } from 'axios'
import type { AxiosResponse } from 'axios'
import type { DetectionResponse, ApiError } from './types'

// Create Axios instance with base URL from environment
export const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || '',
  timeout: 30000, // 30 seconds
  headers: {
    'Content-Type': 'application/json'
  }
})

// Create Axios instance for file uploads (multipart/form-data)
export const uploadClient = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || '',
  timeout: 60000, // 60 seconds for video uploads
  headers: {
    'Content-Type': 'multipart/form-data'
  }
})

// Request interceptor - add auth tokens or logging in future
apiClient.interceptors.request.use(
  (config) => {
    console.log(`[API] ${config.method?.toUpperCase()} ${config.url}`)
    return config
  },
  (error: AxiosError) => {
    console.error('[API] Request error:', error)
    return Promise.reject(error)
  }
)

uploadClient.interceptors.request.use(
  (config) => {
    console.log(`[Upload] ${config.method?.toUpperCase()} ${config.url}`)
    return config
  },
  (error: AxiosError) => {
    console.error('[Upload] Request error:', error)
    return Promise.reject(error)
  }
)

// Response interceptor - centralized error handling
apiClient.interceptors.response.use(
  (response: AxiosResponse) => {
    console.log(`[API] Response:`, response.status, response.data)
    return response
  },
  (error: AxiosError<ApiError>) => {
    console.error('[API] Response error:', error.response?.data || error.message)

    // Handle common errors
    if (error.response?.status === 404) {
      console.error('[API] Endpoint not found')
    } else if (error.response?.status === 500) {
      console.error('[API] Server error')
    } else if (error.code === 'ECONNABORTED') {
      console.error('[API] Request timeout')
    }

    return Promise.reject(error)
  }
)

uploadClient.interceptors.response.use(
  (response: AxiosResponse) => {
    console.log(`[Upload] Response:`, response.status, response.data)
    return response
  },
  (error: AxiosError<ApiError>) => {
    console.error('[Upload] Response error:', error.response?.data || error.message)
    return Promise.reject(error)
  }
)

// Type-safe API request helpers
export async function detectImage(file: File): Promise<DetectionResponse> {
  const formData = new FormData()
  formData.append('file', file)

  const response = await uploadClient.post<DetectionResponse>(
    '/detect/image',
    formData
  )

  return response.data
}

export async function detectVideo(file: File): Promise<DetectionResponse> {
  const formData = new FormData()
  formData.append('file', file)

  const response = await uploadClient.post<DetectionResponse>(
    '/detect/video',
    formData
  )

  return response.data
}
