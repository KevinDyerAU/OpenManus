import { useState, useEffect, useRef } from 'react'
import {
  Send,
  Bot,
  User,
  Settings,
  MessageSquare,
  Globe,
  Brain,
  Code,
  Activity,
  Trash2,
  Download,
  Upload,
  Moon,
  Sun,
  Loader2,
  CheckCircle,
  XCircle,
  AlertCircle,
  Zap,
  Clock,
  TrendingUp,
  Play,
  Pause
} from 'lucide-react'
import { Button } from '@/components/ui/button.jsx'
import { Input } from '@/components/ui/input.jsx'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card.jsx'
import { Badge } from '@/components/ui/badge.jsx'
import { ScrollArea } from '@/components/ui/scroll-area.jsx'
import { Separator } from '@/components/ui/separator.jsx'
import { Textarea } from '@/components/ui/textarea.jsx'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select.jsx'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs.jsx'
import { Switch } from '@/components/ui/switch.jsx'
import { Label } from '@/components/ui/label.jsx'
import './App.css'

// Configuration
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws/chat'

// Available models and task types
const MODELS = [
  { id: 'auto', name: 'Auto Select (Recommended)', provider: 'OpenManus' },
  { id: 'lmstudio/deepseek/deepseek-r1-0528-qwen3-8b', name: 'DeepSeek R1 8B (Local)', provider: 'LM Studio' },
  { id: 'openai/gpt-4o', name: 'GPT-4o', provider: 'OpenAI' },
  { id: 'openai/gpt-4o-mini', name: 'GPT-4o Mini', provider: 'OpenAI' },
  { id: 'anthropic/claude-3.5-sonnet', name: 'Claude 3.5 Sonnet', provider: 'Anthropic' },
  { id: 'google/gemini-2.0-flash-exp', name: 'Gemini 2.0 Flash', provider: 'Google' },
  { id: 'openrouter/anthropic/claude-3.5-sonnet', name: 'Claude 3.5 Sonnet (OpenRouter)', provider: 'OpenRouter' },
  { id: 'openrouter/openai/gpt-4o', name: 'GPT-4o (OpenRouter)', provider: 'OpenRouter' },
  { id: 'openrouter/google/gemini-2.0-flash-exp', name: 'Gemini 2.0 Flash (OpenRouter)', provider: 'OpenRouter' },
  { id: 'openrouter/meta-llama/llama-3.3-70b-instruct', name: 'Llama 3.3 70B (OpenRouter)', provider: 'OpenRouter' },
  { id: 'openrouter/qwen/qwen-2.5-72b-instruct', name: 'Qwen 2.5 72B (OpenRouter)', provider: 'OpenRouter' }
]

const TASK_TYPES = [
  { id: 'general_chat', name: 'General Chat', icon: MessageSquare, description: 'General conversation and Q&A' },
  { id: 'web_browsing', name: 'Web Browsing', icon: Globe, description: 'Browse and search the web' },
  { id: 'manus_agent', name: 'Manus Agent', icon: Brain, description: 'Full agent capabilities with tools' },
  { id: 'code_analysis', name: 'Code Analysis', icon: Code, description: 'Analyze and review code' }
]

function App() {
  // State management
  const [messages, setMessages] = useState([])
  const [inputMessage, setInputMessage] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [connectionStatus, setConnectionStatus] = useState('disconnected')
  const [conversations, setConversations] = useState([])
  const [conversationId, setConversationId] = useState(null)
  const [selectedModel, setSelectedModel] = useState('auto')
  const [selectedTaskType, setSelectedTaskType] = useState('general_chat')
  
  // Debug: Log state changes
  useEffect(() => {
    console.log('selectedTaskType changed to:', selectedTaskType)
  }, [selectedTaskType])
  const [streamingEnabled, setStreamingEnabled] = useState(true)
  const [activeTab, setActiveTab] = useState('chat')
  const [agentThoughts, setAgentThoughts] = useState([])
  const [darkMode, setDarkMode] = useState(() => {
    try {
      // Initialize from localStorage or system preference
      const saved = localStorage.getItem('darkMode')
      if (saved !== null) {
        return JSON.parse(saved)
      }
      // Default to system preference
      if (typeof window !== 'undefined' && window.matchMedia) {
        return window.matchMedia('(prefers-color-scheme: dark)').matches
      }
      return false
    } catch (error) {
      console.warn('Error initializing dark mode:', error)
      return false
    }
  })
  const [taskProgress, setTaskProgress] = useState({})
  const [showProgress, setShowProgress] = useState(false)

  // Callback state
  const [callbacksEnabled, setCallbacksEnabled] = useState(false)
  const [callbackConfig, setCallbackConfig] = useState({
    delivery_method: 'websocket',
    webhook_url: '',
    event_types: ['workflow', 'text', 'thinking', 'tool_use', 'error', 'finish']
  })
  const [callbackSession, setCallbackSession] = useState(null)
  const [callbackEvents, setCallbackEvents] = useState([])
  const [callbackStats, setCallbackStats] = useState({
    total_events: 0,
    successful_deliveries: 0,
    failed_deliveries: 0
  })

  // Refs
  const messagesEndRef = useRef(null)
  const wsRef = useRef(null)
  const inputRef = useRef(null)

  // Auto-scroll to bottom
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  // Dark mode effect
  useEffect(() => {
    try {
      localStorage.setItem('darkMode', JSON.stringify(darkMode))
      if (darkMode) {
        document.documentElement.classList.add('dark')
      } else {
        document.documentElement.classList.remove('dark')
      }
    } catch (error) {
      console.warn('Error saving dark mode preference:', error)
    }
  }, [darkMode])

  // WebSocket connection
  const connectWebSocket = () => {
    try {
      wsRef.current = new WebSocket(WS_URL)

      wsRef.current.onopen = () => {
        setConnectionStatus('connected')
        console.log('WebSocket connected')
      }

      wsRef.current.onmessage = (event) => {
        const data = JSON.parse(event.data)

        if (data.type === 'message') {
          setMessages(prev => [...prev, {
            id: Date.now(),
            type: 'assistant',
            content: data.content,
            timestamp: new Date().toISOString()
          }])
          setIsLoading(false)
          setShowProgress(false)
          setTaskProgress({})
        } else if (data.type === 'progress') {
          // Handle real-time progress updates
          setShowProgress(true)
          setTaskProgress({
            status: data.status,
            message: data.message,
            progress: data.progress || 0
          })
          console.log('Progress update:', data.message, `${data.progress}%`)

          // If this is an agent thought (contains 🤖), add it to the thoughts tab instead of chat
          if (data.message && data.message.includes('🤖')) {
            setAgentThoughts(prev => [{
              id: Date.now() + Math.random(),
              content: data.message,
              timestamp: new Date().toISOString()
            }, ...prev])  // Add to beginning so latest appears at top
          }
        } else if (data.type === 'callback') {
          setCallbackEvents(prev => [...prev, data])
          setCallbackStats(prev => ({
            ...prev,
            total_events: prev.total_events + 1,
            successful_deliveries: prev.successful_deliveries + 1
          }))
        }

        // Only set loading to false for final messages, not progress updates
        if (data.type !== 'progress') {
          setIsLoading(false)
        }
      }

      wsRef.current.onclose = () => {
        setConnectionStatus('disconnected')
        console.log('WebSocket disconnected')
      }

      wsRef.current.onerror = (error) => {
        setConnectionStatus('error')
        console.error('WebSocket error:', error)
      }
    } catch (error) {
      console.error('Failed to connect WebSocket:', error)
      setConnectionStatus('error')
    }
  }

  const disconnectWebSocket = () => {
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
  }

  // Send message
  const sendMessage = async () => {
    if (!inputMessage.trim()) return

    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: inputMessage,
      timestamp: new Date().toISOString()
    }

    setMessages(prev => [...prev, userMessage])
    setIsLoading(true)
    setInputMessage('')

    try {
      if (streamingEnabled && wsRef.current?.readyState === WebSocket.OPEN) {
        // Send via WebSocket
        const wsMessage = {
          type: 'chat',
          message: inputMessage,
          model: selectedModel,
          task_type: selectedTaskType,
          conversation_id: conversationId,
          callback_config: callbacksEnabled ? callbackConfig : null
        }
        console.log('Sending WebSocket message with task_type:', selectedTaskType)
        console.log('Full WebSocket message:', wsMessage)
        wsRef.current.send(JSON.stringify(wsMessage))
      } else {
        // Send via HTTP
        const requestData = {
          message: inputMessage,
          model: selectedModel,
          task_type: selectedTaskType,
          conversation_id: conversationId,
          callback_config: callbacksEnabled ? callbackConfig : null
        }
        console.log('Sending HTTP request with task_type:', selectedTaskType)
        console.log('Full HTTP request data:', requestData)

        const response = await fetch(`${API_BASE_URL}/chat`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(requestData),
        })

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`)
        }

        const data = await response.json()
        setMessages(prev => [...prev, {
          id: Date.now(),
          type: 'assistant',
          content: data.response,
          timestamp: new Date().toISOString()
        }])
        setIsLoading(false)
      }
    } catch (error) {
      console.error('Error sending message:', error)
      setMessages(prev => [...prev, {
        id: Date.now(),
        type: 'error',
        content: `Error: ${error.message}`,
        timestamp: new Date().toISOString()
      }])
      setIsLoading(false)
    }
  }

  // Handle key press
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  // Clear conversation
  const clearConversation = () => {
    setMessages([])
    setAgentThoughts([])
    setConversationId(null)
    setTaskProgress({})
    setShowProgress(false)
  }

  // Connect WebSocket on mount
  useEffect(() => {
    if (streamingEnabled) {
      connectWebSocket()
    }
    return () => {
      disconnectWebSocket()
    }
  }, [streamingEnabled])

  return (
    <div className={`min-h-screen ${darkMode ? 'dark' : ''}`}>
      <div className="flex h-screen bg-background text-foreground">
        {/* Main Chat Area */}
        <div className="flex-1 flex flex-col">
          {/* Header */}
          <div className="border-b bg-card">
            <div className="flex items-center justify-between p-4">
              <div className="flex items-center space-x-3">
                <div className="flex items-center space-x-2">
                  <Bot className="h-8 w-8 text-primary" />
                  <h1 className="text-2xl font-bold">OpenManus</h1>
                </div>
                <Badge variant={connectionStatus === 'connected' ? 'default' : 'destructive'}>
                  {connectionStatus}
                </Badge>
              </div>

              <div className="flex items-center space-x-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setDarkMode(!darkMode)}
                >
                  {darkMode ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
                </Button>
              </div>
            </div>
          </div>

          {/* Chat Interface */}
          <div className="flex-1 flex">
            {/* Chat Messages */}
            <div className="flex-1 flex flex-col">
              <Card className="flex-1 m-4 flex flex-col">
                <CardHeader className="pb-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <CardTitle className="text-lg">Chat</CardTitle>
                      <Badge variant="outline">
                        {messages.length} messages
                      </Badge>
                      <Badge variant="secondary">
                        {TASK_TYPES.find(t => t.id === selectedTaskType)?.name || 'Unknown'}
                      </Badge>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Badge variant={streamingEnabled ? 'default' : 'secondary'}>
                        {streamingEnabled ? 'Streaming' : 'HTTP'}
                      </Badge>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={clearConversation}
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                </CardHeader>

                <CardContent className="flex-1 flex flex-col min-h-0">
                  {/* Agent Thoughts Section (if any) */}
                  {agentThoughts && agentThoughts.length > 0 && (
                    <div className="flex-shrink-0 mb-3 border-b border-border pb-3">
                      <div className="flex items-center space-x-2 mb-2">
                        <Bot className="h-4 w-4 text-purple-600" />
                        <h3 className="text-xs font-medium text-purple-600">Agent Thoughts ({agentThoughts.length})</h3>
                      </div>
                      <ScrollArea className="h-20 pr-2">
                        <div className="space-y-1">
                          {agentThoughts
                            .filter(thought => thought && thought.content)
                            .slice(0, 3)
                            .map((thought, index) => (
                              <div key={thought.id || `thought-${index}`} className="text-xs p-2 bg-purple-50 dark:bg-purple-900/20 text-purple-800 dark:text-purple-200 rounded border border-purple-200 dark:border-purple-700">
                                <p className="whitespace-pre-wrap break-words leading-tight m-0 text-xs">
                                  {thought.content || 'No content'}
                                </p>
                                <p className="text-xs opacity-70 mt-1">
                                  {thought.timestamp ? new Date(thought.timestamp).toLocaleTimeString() : 'Unknown time'}
                                </p>
                              </div>
                            ))
                          }
                          {agentThoughts.length > 3 && (
                            <p className="text-xs text-muted-foreground text-center py-1">
                              ... and {agentThoughts.length - 3} more thoughts
                            </p>
                          )}
                        </div>
                      </ScrollArea>
                    </div>
                  )}

                  {/* Messages */}
                  <ScrollArea className="flex-1 pr-4 min-h-0">
                    <div className="space-y-4">
                      {messages.length === 0 ? (
                        <div className="text-center text-muted-foreground py-8">
                          <Bot className="h-12 w-12 mx-auto mb-4 opacity-50" />
                          <p>Start a conversation with OpenManus</p>
                          <p className="text-sm">Choose a task type and model, then send a message</p>
                        </div>
                      ) : (
                        messages.map((message) => (
                          <div
                            key={message.id}
                            className={`flex items-start space-x-3 ${message.type === 'user' ? 'justify-end' : 'justify-start'
                              }`}
                          >
                            {message.type !== 'user' && (
                              <div className="flex-shrink-0">
                                {message.type === 'error' ? (
                                  <AlertCircle className="h-6 w-6 text-destructive" />
                                ) : (
                                  <Bot className="h-6 w-6 text-primary" />
                                )}
                              </div>
                            )}

                            <div
                              className={`${message.type === 'user' ? 'max-w-[80%]' : 'max-w-[90%]'} rounded-lg p-3 chat-message ${message.type === 'user'
                                  ? 'bg-primary text-primary-foreground'
                                  : message.type === 'error'
                                    ? 'bg-destructive/10 text-destructive border border-destructive/20'
                                    : 'bg-muted'
                                }`}
                            >
                              <div className="message-content">
                                <p className="whitespace-pre-wrap break-words leading-relaxed m-0">{message.content}</p>
                              </div>
                              <p className="text-xs opacity-70 mt-1">
                                {new Date(message.timestamp).toLocaleTimeString()}
                              </p>
                            </div>

                            {message.type === 'user' && (
                              <div className="flex-shrink-0">
                                <User className="h-6 w-6 text-primary" />
                              </div>
                            )}
                          </div>
                        ))
                      )}

                      {isLoading && (
                        <div className="flex items-start space-x-3">
                          <Bot className="h-6 w-6 text-primary" />
                          <div className="bg-muted rounded-lg p-3">
                            <div className="flex items-center space-x-2">
                              <Loader2 className="h-4 w-4 animate-spin" />
                              <span>Thinking...</span>
                            </div>
                          </div>
                        </div>
                      )}

                      {showProgress && taskProgress && (
                        <div className="flex items-start space-x-3">
                          <Bot className="h-6 w-6 text-primary" />
                          <div className="bg-muted rounded-lg p-3 w-full max-w-md">
                            <div className="space-y-2">
                              <div className="flex items-center space-x-2">
                                <Loader2 className="h-4 w-4 animate-spin" />
                                <span className="text-sm font-medium">Task Progress</span>
                              </div>
                              <div className="text-xs text-muted-foreground">
                                {taskProgress.message}
                              </div>
                              {taskProgress.progress > 0 && (
                                <div className="w-full bg-background rounded-full h-2">
                                  <div
                                    className="bg-primary h-2 rounded-full transition-all duration-300"
                                    style={{ width: `${taskProgress.progress}%` }}
                                  ></div>
                                </div>
                              )}
                              <div className="text-xs text-muted-foreground">
                                Status: {taskProgress.status} {taskProgress.progress > 0 && `(${taskProgress.progress}%)`}
                              </div>
                            </div>
                          </div>
                        </div>
                      )}

                      <div ref={messagesEndRef} />
                    </div>
                  </ScrollArea>

                  {/* Input Area */}
                  <div className="pt-4 border-t">
                    <div className="flex space-x-2">
                      <Textarea
                        ref={inputRef}
                        value={inputMessage}
                        onChange={(e) => setInputMessage(e.target.value)}
                        onKeyPress={handleKeyPress}
                        placeholder="Type your message..."
                        className="flex-1 min-h-[60px] resize-none"
                        disabled={isLoading}
                      />
                      <Button
                        onClick={sendMessage}
                        disabled={isLoading || !inputMessage.trim()}
                        size="lg"
                      >
                        <Send className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Sidebar */}
            <div className="w-80 p-4 border-l bg-card/50">
              <div className="space-y-4">
                {/* Model Selection */}
                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm">Model</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <Select value={selectedModel} onValueChange={setSelectedModel}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {MODELS.map((model) => (
                          <SelectItem key={model.id} value={model.id}>
                            <div className="flex flex-col">
                              <span>{model.name}</span>
                              <span className="text-xs text-muted-foreground">{model.provider}</span>
                            </div>
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </CardContent>
                </Card>

                {/* Task Type Selection */}
                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm">Task Type</CardTitle>
                    {/* Debug: Show current state */}
                    <div className="text-xs text-muted-foreground">
                      Current: {selectedTaskType}
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2">
                      {TASK_TYPES.map((taskType) => {
                        const Icon = taskType.icon
                        const isSelected = selectedTaskType === taskType.id
                        console.log(`Task ${taskType.id}: selected=${isSelected}, selectedTaskType=${selectedTaskType}`)
                        return (
                          <div
                            key={taskType.id}
                            className={`p-3 rounded-lg border cursor-pointer transition-all duration-200 ${
                              isSelected
                                ? 'border-primary bg-primary/20 shadow-md ring-2 ring-primary/30'
                                : 'border-border hover:bg-muted hover:border-muted-foreground/20'
                            }`}
                            style={{
                              backgroundColor: isSelected ? 'rgba(59, 130, 246, 0.15)' : 'transparent',
                              borderColor: isSelected ? '#3b82f6' : '#e5e7eb',
                              boxShadow: isSelected ? '0 4px 6px -1px rgba(0, 0, 0, 0.1)' : 'none'
                            }}
                            onClick={(e) => {
                              e.preventDefault()
                              e.stopPropagation()
                              console.log('Click event fired for task:', taskType.id)
                              console.log('Current selectedTaskType before update:', selectedTaskType)
                              setSelectedTaskType(taskType.id)
                              console.log('setSelectedTaskType called with:', taskType.id)
                            }}
                          >
                            <div className="flex items-center space-x-3">
                              <Icon className="h-5 w-5" />
                              <div className="flex-1">
                                <div className="font-medium text-sm">{taskType.name}</div>
                                <div className="text-xs text-muted-foreground">{taskType.description}</div>
                              </div>
                            </div>
                          </div>
                        )
                      })}
                    </div>
                  </CardContent>
                </Card>

                {/* Settings */}
                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm">Settings</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="flex items-center justify-between">
                      <Label htmlFor="streaming" className="text-sm">Real-time Streaming</Label>
                      <Switch
                        id="streaming"
                        checked={streamingEnabled}
                        onCheckedChange={setStreamingEnabled}
                      />
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
