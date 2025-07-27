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
  { id: 'openai/gpt-4o', name: 'GPT-4o', provider: 'OpenAI' },
  { id: 'openai/gpt-4o-mini', name: 'GPT-4o Mini', provider: 'OpenAI' },
  { id: 'anthropic/claude-3.5-sonnet', name: 'Claude 3.5 Sonnet', provider: 'Anthropic' },
  { id: 'google/gemini-2.0-flash-exp', name: 'Gemini 2.0 Flash', provider: 'Google' },
  { id: 'meta-llama/llama-3.1-405b', name: 'Llama 3.1 405B', provider: 'Meta' },
]

const TASK_TYPES = [
  { id: 'general_chat', name: 'General Chat', icon: MessageSquare },
  { id: 'code_generation', name: 'Code Generation', icon: Code },
  { id: 'data_analysis', name: 'Data Analysis', icon: Activity },
  { id: 'web_browsing', name: 'Web Browsing', icon: Globe },
  { id: 'reasoning', name: 'Reasoning', icon: Brain },
  { id: 'creative_writing', name: 'Creative Writing', icon: Zap },
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
  const [streamingEnabled, setStreamingEnabled] = useState(true)
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
  const [activeTab, setActiveTab] = useState('chat')
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

  // Scroll to bottom of messages
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  // WebSocket connection
  useEffect(() => {
    if (streamingEnabled) {
      connectWebSocket()
    } else {
      disconnectWebSocket()
    }

    return () => disconnectWebSocket()
  }, [streamingEnabled])

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
        } else if (data.type === 'callback') {
          setCallbackEvents(prev => [...prev, data])
          setCallbackStats(prev => ({
            ...prev,
            total_events: prev.total_events + 1,
            successful_deliveries: prev.successful_deliveries + 1
          }))
        }
        
        setIsLoading(false)
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

    setIsLoading(true)
    setInputMessage('')

    try {
      if (streamingEnabled && wsRef.current?.readyState === WebSocket.OPEN) {
        // Send via WebSocket
        wsRef.current.send(JSON.stringify({
          type: 'chat',
          message: inputMessage,
          model: selectedModel,
          task_type: selectedTaskType,
          conversation_id: conversationId,
          callback_config: callbacksEnabled ? callbackConfig : null
        }))
      } else {
        // Send via HTTP
        const requestData = {
          message: inputMessage,
          model: selectedModel,
          task_type: selectedTaskType,
          conversation_id: conversationId,
          callback_config: callbacksEnabled ? callbackConfig : null
        }
        
        console.log('DEBUG: Sending request with task_type:', selectedTaskType)
        console.log('DEBUG: Full request data:', requestData)
        
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
        
        console.log('DEBUG: Received response data:', data)
        console.log('DEBUG: Response task_type:', data.task_type)
        
        setMessages(prev => [...prev, {
          id: Date.now() + 1,
          type: 'assistant',
          content: data.response,
          timestamp: new Date().toISOString()
        }])

        if (data.conversation_id) {
          setConversationId(data.conversation_id)
        }

        setIsLoading(false)
      }
    } catch (error) {
      console.error('Error sending message:', error)
      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        type: 'error',
        content: 'Failed to send message. Please try again.',
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
    setConversationId(null)
    setCallbackEvents([])
    setCallbackStats({
      total_events: 0,
      successful_deliveries: 0,
      failed_deliveries: 0
    })
  }

  // Initialize dark mode on component mount
  useEffect(() => {
    try {
      // Apply dark class based on initial state
      if (darkMode) {
        document.documentElement.classList.add('dark')
        console.log('Dark mode applied on mount')
      } else {
        document.documentElement.classList.remove('dark')
        console.log('Light mode applied on mount')
      }
    } catch (error) {
      console.error('Error applying dark mode on mount:', error)
    }
  }, [darkMode])

  // Toggle dark mode
  const toggleDarkMode = () => {
    try {
      const newDarkMode = !darkMode
      console.log('Toggling dark mode to:', newDarkMode)
      
      setDarkMode(newDarkMode)
      
      // Apply/remove dark class
      if (newDarkMode) {
        document.documentElement.classList.add('dark')
        console.log('Dark class added to document')
      } else {
        document.documentElement.classList.remove('dark')
        console.log('Dark class removed from document')
      }
      
      // Verify DOM state
      console.log('Document classes:', document.documentElement.className)
      console.log('Has dark class:', document.documentElement.classList.contains('dark'))
      
      // Persist preference
      localStorage.setItem('darkMode', JSON.stringify(newDarkMode))
      console.log('Dark mode preference saved:', newDarkMode)
    } catch (error) {
      console.error('Error toggling dark mode:', error)
    }
  }

  // Poll for task progress
  const pollProgress = async (convId) => {
    try {
      const response = await fetch(`${API_BASE_URL}/progress/${convId}`)
      if (response.ok) {
        const progress = await response.json()
        setTaskProgress(progress)
        
        // Show progress if task is processing
        if (progress.status === 'processing') {
          setShowProgress(true)
          // Continue polling
          setTimeout(() => pollProgress(convId), 1000)
        } else if (progress.status === 'completed') {
          setShowProgress(false)
          setTaskProgress({})
        }
      }
    } catch (error) {
      console.error('Error polling progress:', error)
    }
  }

  // Create callback session
  const createCallbackSession = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/callbacks/sessions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(callbackConfig),
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const session = await response.json()
      setCallbackSession(session)
      setCallbacksEnabled(true)
    } catch (error) {
      console.error('Error creating callback session:', error)
    }
  }

  // Get connection status color
  const getConnectionStatusColor = () => {
    switch (connectionStatus) {
      case 'connected': return 'text-green-500'
      case 'disconnected': return 'text-gray-500'
      case 'error': return 'text-red-500'
      default: return 'text-gray-500'
    }
  }

  // Get connection status icon
  const getConnectionStatusIcon = () => {
    switch (connectionStatus) {
      case 'connected': return <CheckCircle className="h-4 w-4" />
      case 'disconnected': return <XCircle className="h-4 w-4" />
      case 'error': return <AlertCircle className="h-4 w-4" />
      default: return <Clock className="h-4 w-4" />
    }
  }

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="container mx-auto max-w-6xl p-4">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center space-x-3">
            <Bot className="h-8 w-8 text-primary" />
            <div>
              <h1 className="text-2xl font-bold">OpenManus Chat</h1>
              <p className="text-sm text-muted-foreground">AI Agent Platform</p>
            </div>
          </div>
          
          <div className="flex items-center space-x-2">
            <div className={`flex items-center space-x-1 ${getConnectionStatusColor()}`}>
              {getConnectionStatusIcon()}
              <span className="text-sm capitalize">{connectionStatus}</span>
            </div>
            
            <Button
              variant="outline"
              size="sm"
              onClick={toggleDarkMode}
            >
              {darkMode ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
            </Button>
          </div>
        </div>

        {/* Main Content */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="chat">Chat</TabsTrigger>
            <TabsTrigger value="settings">Settings</TabsTrigger>
            <TabsTrigger value="callbacks">Callbacks</TabsTrigger>
            <TabsTrigger value="analytics">Analytics</TabsTrigger>
          </TabsList>

          {/* Chat Tab */}
          <TabsContent value="chat" className="space-y-4">
            <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
              {/* Chat Area */}
              <div className="lg:col-span-3">
                <Card className="min-h-[600px] max-h-[80vh] flex flex-col">
                  <CardHeader className="pb-3">
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-lg">Conversation</CardTitle>
                      <div className="flex items-center space-x-2">
                        <Badge variant="outline">
                          {messages.filter(m => m.type === 'user').length} messages
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
                  
                  <CardContent className="flex-1 flex flex-col">
                    {/* Messages */}
                    <ScrollArea className="flex-1 pr-4">
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
                              className={`flex items-start space-x-3 ${
                                message.type === 'user' ? 'justify-end' : 'justify-start'
                              }`}
                            >
                              {message.type !== 'user' && (
                                <div className="flex-shrink-0">
                                  {message.type === 'assistant' ? (
                                    <Bot className="h-6 w-6 text-primary" />
                                  ) : (
                                    <AlertCircle className="h-6 w-6 text-destructive" />
                                  )}
                                </div>
                              )}
                              
                              <div
                                className={`${message.type === 'user' ? 'max-w-[80%]' : 'max-w-[90%]'} rounded-lg p-3 chat-message ${
                                  message.type === 'user'
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

                {/* Task Type */}
                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm">Task Type</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 gap-2">
                      {TASK_TYPES.map((task) => {
                        const Icon = task.icon
                        return (
                          <Button
                            key={task.id}
                            variant={selectedTaskType === task.id ? "default" : "outline"}
                            size="sm"
                            onClick={() => setSelectedTaskType(task.id)}
                            className="h-auto p-2 flex flex-col items-center space-y-1"
                          >
                            <Icon className="h-4 w-4" />
                            <span className="text-xs">{task.name}</span>
                          </Button>
                        )
                      })}
                    </div>
                  </CardContent>
                </Card>

                {/* Quick Settings */}
                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm">Quick Settings</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="flex items-center justify-between">
                      <Label htmlFor="streaming" className="text-sm">Streaming</Label>
                      <Switch
                        id="streaming"
                        checked={streamingEnabled}
                        onCheckedChange={setStreamingEnabled}
                      />
                    </div>
                    
                    <div className="flex items-center justify-between">
                      <Label htmlFor="callbacks" className="text-sm">Callbacks</Label>
                      <Switch
                        id="callbacks"
                        checked={callbacksEnabled}
                        onCheckedChange={setCallbacksEnabled}
                      />
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>
          </TabsContent>

          {/* Settings Tab */}
          <TabsContent value="settings" className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <Card>
                <CardHeader>
                  <CardTitle>API Configuration</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <Label htmlFor="api-url">API Base URL</Label>
                    <Input
                      id="api-url"
                      value={API_BASE_URL}
                      placeholder="http://localhost:8000"
                      readOnly
                    />
                  </div>
                  
                  <div>
                    <Label htmlFor="ws-url">WebSocket URL</Label>
                    <Input
                      id="ws-url"
                      value={WS_URL}
                      placeholder="ws://localhost:8000/ws/chat"
                      readOnly
                    />
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Preferences</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-center justify-between">
                    <Label htmlFor="dark-mode">Dark Mode</Label>
                    <Switch
                      id="dark-mode"
                      checked={darkMode}
                      onCheckedChange={toggleDarkMode}
                    />
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <Label htmlFor="auto-scroll">Auto Scroll</Label>
                    <Switch
                      id="auto-scroll"
                      checked={true}
                      disabled
                    />
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Callbacks Tab */}
          <TabsContent value="callbacks" className="space-y-4">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <Card>
                <CardHeader>
                  <CardTitle>Callback Configuration</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <Label htmlFor="delivery-method">Delivery Method</Label>
                    <Select
                      value={callbackConfig.delivery_method}
                      onValueChange={(value) => setCallbackConfig(prev => ({ ...prev, delivery_method: value }))}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="websocket">WebSocket</SelectItem>
                        <SelectItem value="webhook">Webhook</SelectItem>
                        <SelectItem value="sse">Server-Sent Events</SelectItem>
                        <SelectItem value="polling">Polling</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  {callbackConfig.delivery_method === 'webhook' && (
                    <div>
                      <Label htmlFor="webhook-url">Webhook URL</Label>
                      <Input
                        id="webhook-url"
                        value={callbackConfig.webhook_url}
                        onChange={(e) => setCallbackConfig(prev => ({ ...prev, webhook_url: e.target.value }))}
                        placeholder="https://your-webhook-url.com"
                      />
                    </div>
                  )}

                  <div className="space-y-2">
                    <Label>Event Types</Label>
                    <div className="grid grid-cols-2 gap-2">
                      {['workflow', 'text', 'thinking', 'tool_use', 'error', 'finish'].map((eventType) => (
                        <div key={eventType} className="flex items-center space-x-2">
                          <Switch
                            id={eventType}
                            checked={callbackConfig.event_types.includes(eventType)}
                            onCheckedChange={(checked) => {
                              if (checked) {
                                setCallbackConfig(prev => ({
                                  ...prev,
                                  event_types: [...prev.event_types, eventType]
                                }))
                              } else {
                                setCallbackConfig(prev => ({
                                  ...prev,
                                  event_types: prev.event_types.filter(t => t !== eventType)
                                }))
                              }
                            }}
                          />
                          <Label htmlFor={eventType} className="text-sm">{eventType}</Label>
                        </div>
                      ))}
                    </div>
                  </div>

                  <Button onClick={createCallbackSession} className="w-full">
                    Create Callback Session
                  </Button>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Callback Events</CardTitle>
                </CardHeader>
                <CardContent>
                  <ScrollArea className="h-[400px]">
                    <div className="space-y-2">
                      {callbackEvents.length === 0 ? (
                        <p className="text-muted-foreground text-center py-8">
                          No callback events yet
                        </p>
                      ) : (
                        callbackEvents.map((event, index) => (
                          <div key={index} className="p-2 border rounded-lg">
                            <div className="flex items-center justify-between">
                              <Badge variant="outline">{event.event_type}</Badge>
                              <span className="text-xs text-muted-foreground">
                                {new Date(event.timestamp).toLocaleTimeString()}
                              </span>
                            </div>
                            <p className="text-sm mt-1">{event.data?.message || 'Event triggered'}</p>
                          </div>
                        ))
                      )}
                    </div>
                  </ScrollArea>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Analytics Tab */}
          <TabsContent value="analytics" className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Total Messages</CardTitle>
                  <MessageSquare className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{messages.length}</div>
                  <p className="text-xs text-muted-foreground">
                    {messages.filter(m => m.type === 'user').length} sent, {messages.filter(m => m.type === 'assistant').length} received
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Callback Events</CardTitle>
                  <Activity className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{callbackStats.total_events}</div>
                  <p className="text-xs text-muted-foreground">
                    {callbackStats.successful_deliveries} successful
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Connection Status</CardTitle>
                  <TrendingUp className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold capitalize">{connectionStatus}</div>
                  <p className="text-xs text-muted-foreground">
                    {streamingEnabled ? 'Streaming enabled' : 'HTTP mode'}
                  </p>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}

export default App

