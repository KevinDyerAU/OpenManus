import React from 'react'
import { Loader2, Bot } from 'lucide-react'
import { Card, CardContent } from '@/components/ui/card.jsx'

/**
 * LoadingState component for displaying various loading states throughout the application
 * @param {Object} props - Component props
 * @param {string} props.message - Loading message to display
 * @param {string} props.variant - Loading variant: 'default', 'minimal', 'chat', 'fullscreen'
 * @param {boolean} props.showIcon - Whether to show loading icon
 * @param {string} props.className - Additional CSS classes
 */
export const LoadingState = ({ 
  message = "Loading...", 
  variant = "default", 
  showIcon = true,
  className = ""
}) => {
  const baseClasses = "flex items-center justify-center"
  
  if (variant === "minimal") {
    return (
      <div className={`${baseClasses} p-2 ${className}`}>
        {showIcon && <Loader2 className="w-4 h-4 animate-spin mr-2" />}
        <span className="text-sm text-gray-600 dark:text-gray-400">{message}</span>
      </div>
    )
  }
  
  if (variant === "chat") {
    return (
      <div className={`${baseClasses} p-4 ${className}`}>
        <div className="flex items-center space-x-2">
          <Bot className="w-5 h-5 text-blue-500" />
          <div className="flex space-x-1">
            <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"></div>
            <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
            <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
          </div>
        </div>
      </div>
    )
  }
  
  if (variant === "fullscreen") {
    return (
      <div className={`min-h-screen ${baseClasses} flex-col space-y-4 ${className}`}>
        {showIcon && <Loader2 className="w-8 h-8 animate-spin text-blue-500" />}
        <p className="text-lg text-gray-600 dark:text-gray-400">{message}</p>
      </div>
    )
  }
  
  // Default variant
  return (
    <Card className={className}>
      <CardContent className={`${baseClasses} p-6 space-x-3`}>
        {showIcon && <Loader2 className="w-6 h-6 animate-spin text-blue-500" />}
        <span className="text-gray-600 dark:text-gray-400">{message}</span>
      </CardContent>
    </Card>
  )
}

/**
 * NetworkErrorState component for displaying network-related errors
 * @param {Object} props - Component props
 * @param {Function} props.onRetry - Retry function
 * @param {string} props.message - Error message
 */
export const NetworkErrorState = ({ onRetry, message = "Network connection failed" }) => {
  return (
    <Card className="border-red-200 dark:border-red-800">
      <CardContent className="flex items-center justify-center p-6 space-x-3">
        <div className="text-red-500">
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        </div>
        <div className="flex-1">
          <p className="text-red-600 dark:text-red-400">{message}</p>
        </div>
        {onRetry && (
          <button
            onClick={onRetry}
            className="px-3 py-1 text-sm bg-red-100 hover:bg-red-200 dark:bg-red-900/20 dark:hover:bg-red-900/30 text-red-700 dark:text-red-300 rounded transition-colors"
          >
            Retry
          </button>
        )}
      </CardContent>
    </Card>
  )
}

export default LoadingState
