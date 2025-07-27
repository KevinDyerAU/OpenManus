import { useState, useEffect, useRef } from 'react'
import { Button } from '@/components/ui/button.jsx'
import { Input } from '@/components/ui/input.jsx'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card.jsx'
import { Badge } from '@/components/ui/badge.jsx'
import { ScrollArea } from '@/components/ui/scroll-area.jsx'
import { Separator } from '@/components/ui/separator.jsx'
import { Textarea } from '@/components/ui/textarea.jsx'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select.jsx'
import { Switch } from '@/components/ui/switch.jsx'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs.jsx'
import { 
  Send, 
  Bot, 
  User, 
  Settings, 
  MessageSquare, 
  Globe, 
  Code, 
  Brain,
  Zap,
  Clock,
  DollarSign,
  Activity,
  Trash2,
  Download,
  Upload,
  Moon,
  Sun,
  Loader2,
  AlertCircle,
  CheckCircle,
  XCircle
} from 'lucide-react'
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
  { id: 'google/gemini-2.5-pro', name: 'Gemini 2.5 Pro', provider: 'Google' },
  { id: 'meta-llama/llama-3.1-405b', name: 'Llama 3.1 405B', provider: 'Meta' }
]

const TASK_TYPES = [
  { id: 'general_chat', name: 'General Chat', icon: MessageSquare },
  { id: 'code_generation', name: 'Code Generation', icon: Code },
  { id: 'data_analysis', name: 'Data Analysis', icon: Activity },
  { id: 'web_browsing', name: 'Web Browsing', icon: Globe },
  { id: 'reasoning', name: 'Reasoning', icon: Brain },
  { id: 'creative_writing', name: 'Creative Writing', icon: Zap }
]

function App() {
  // State management
  const [messages, setMessages] = useState([])
  const [inputMessage, setInputMessage] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [conversationId, setConversationId] = useState(null)
  const [selectedModel, setSelectedModel] = useState('auto')
  const [selectedTaskType, setSelectedTaskType] = useState('general_chat')
  const [streamingEnabled, setStreamingEnabled] = useState(true)
  const [darkMode, setDarkMode] = useState(false)
  const [connectionStatus, setConnectionStatus] = useState('disconnected')
  const [stats, setStats] = useState({ tokens: 0, cost: 0, requests: 0 })
  const [conversations, setConversations] = useState([])
  const [activeTab, setActiveTab] = useState('chat')
  
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
        
        if (data.type === 'content') {
          // Update the last message with streaming content
          setMessages(prev => {
            const newMessages = [...prev]
            const lastMessage = newMessages[newMessages.length - 1]
            
            if (lastMessage && lastMessage.role === 'assistant' && lastMessage.streaming) {
              lastMessage.content += data.content
            } else {
              newMessages.push({
                id: Date.now(),
                role: 'assistant',
                content: data.content,
                timestamp: new Date(),
                streaming: true,
                conversationId: data.conversation_id
              })
            }
            
            return newMessages
          })
        } else if (data.type === 'done') {
          // Mark streaming as complete
          setMessages(prev => {
            const newMessages = [...prev]
            const lastMessage = newMessages[newMessages.length - 1]
            if (lastMessage && lastMessage.streaming) {
              lastMessage.streaming = false
            }
            return newMessages
          })
          setIsLoading(false)
          setConversationId(data.conversation_id)
        } else if (data.type === 'error') {
          console.error('WebSocket error:', data.error)
          addSystemMessage(`Error: ${data.error}`, 'error')
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
    setConnectionStatus('disconnected')
  }
  
  // Add system message
  const addSystemMessage = (content, type = 'info') => {
    setMessages(prev => [...prev, {
      id: Date.now(),
      role: 'system',
      content,
      timestamp: new Date(),
      type
    }])
  }
  
  // Send message
  const sendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return
    
    const userMessage = {
      id: Date.now(),
      role: 'user',
      content: inputMessage.trim(),
      timestamp: new Date()
    }
    
    setMessages(prev => [...prev, userMessage])
    setIsLoading(true)
    
    const messageToSend = inputMessage.trim()
    setInputMessage('')
    
    try {
      if (streamingEnabled && wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        // Send via WebSocket for streaming
        wsRef.current.send(JSON.stringify({
          type: 'chat',
          message: messageToSend,
          conversation_id: conversationId,
          model: selectedModel === 'auto' ? null : selectedModel,
          task_type: selectedTaskType
        }))
      } else {
        // Send via REST API
        const response = await fetch(`${API_BASE_URL}/chat`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            message: messageToSend,
            conversation_id: conversationId,
            model: selectedModel === 'auto' ? null : selectedModel,
            task_type: selectedTaskType,
            stream: false
          })
        })
        
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`)
        }
        
        const data = await response.json()
        
        const assistantMessage = {
          id: Date.now() + 1,
          role: 'assistant',
          content: data.response,
          timestamp: new Date(),
          model: data.model_used,
          tokens: data.tokens_used,
          cost: data.cost,
          conversationId: data.conversation_id
        }
        
        setMessages(prev => [...prev, assistantMessage])
        setConversationId(data.conversation_id)
        
        // Update stats
        setStats(prev => ({
          tokens: prev.tokens + data.tokens_used,
          cost: prev.cost + data.cost,
          requests: prev.requests + 1
        }))
        
        setIsLoading(false)
      }
    } catch (error) {
      console.error('Error sending message:', error)
      addSystemMessage(`Failed to send message: ${error.message}`, 'error')
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
  
  // Load conversations
  const loadConversations = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/conversations`)
      if (response.ok) {
        const data = await response.json()
        setConversations(data.conversations)
      }
    } catch (error) {
      console.error('Error loading conversations:', error)
    }
  }
  
  // Load conversation
  const loadConversation = async (convId) => {
    try {
      const response = await fetch(`${API_BASE_URL}/conversations/${convId}`)
      if (response.ok) {
        const data = await response.json()
        const loadedMessages = data.messages.map((msg, index) => ({
          id: index,
          role: msg.role,
          content: msg.content,
          timestamp: new Date(msg.timestamp),
          model: msg.model,
          tokens: msg.tokens
        }))
        setMessages(loadedMessages)
        setConversationId(convId)
        setActiveTab('chat')
      }
    } catch (error) {
      console.error('Error loading conversation:', error)
    }
  }
  
  // Clear conversation
  const clearConversation = () => {
    setMessages([])
    setConversationId(null)
    addSystemMessage('New conversation started', 'info')
  }
  
  // Delete conversation
  const deleteConversation = async (convId) => {
    try {
      const response = await fetch(`${API_BASE_URL}/conversations/${convId}`, {
        method: 'DELETE'
      })
      if (response.ok) {
        loadConversations()
        if (conversationId === convId) {
          clearConversation()
        }
      }
    } catch (error) {
      console.error('Error deleting conversation:', error)
    }
  }
  
  // Export conversation
  const exportConversation = () => {
    const exportData = {
      conversation_id: conversationId,
      messages: messages.filter(msg => msg.role !== 'system'),
      exported_at: new Date().toISOString(),
      stats
    }
    
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `conversation_${conversationId || 'current'}.json`
    a.click()
    URL.revokeObjectURL(url)
  }
  
  // Toggle dark mode
  const toggleDarkMode = () => {
    setDarkMode(!darkMode)
    document.documentElement.classList.toggle('dark')
  }
  
  // Load conversations on mount
  useEffect(() => {
    loadConversations()
    addSystemMessage('Welcome to OpenManus Chat! Start a conversation with our AI agents.', 'info')
  }, [])
  
  // Get connection status color
  const getConnectionStatusColor = () => {
    switch (connectionStatus) {
      case 'connected': return 'text-green-500'
      case 'error': return 'text-red-500'
      default: return 'text-yellow-500'
    }
  }
  
  // Get connection status icon
  const getConnectionStatusIcon = () => {
    switch (connectionStatus) {
      case 'connected': return <CheckCircle className="w-4 h-4" />
      case 'error': return <XCircle className="w-4 h-4" />
      default: return <AlertCircle className="w-4 h-4" />
    }
  }
  
  // Format message timestamp
  const formatTimestamp = (timestamp) => {
    return timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  }
  
  // Get task type icon
  const getTaskTypeIcon = (taskType) => {
    const task = TASK_TYPES.find(t => t.id === taskType)
    return task ? task.icon : MessageSquare
  }

  return (
    <div className={`min-h-screen bg-background text-foreground ${darkMode ? 'dark' : ''}`}>
      <div className="container mx-auto max-w-7xl p-4">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <Bot className="w-8 h-8 text-primary" />
              <h1 className="text-2xl font-bold">OpenManus Chat</h1>
            </div>
            <Badge variant="outline" className="flex items-center space-x-1">
              <div className={getConnectionStatusColor()}>
                {getConnectionStatusIcon()}
              </div>
              <span className="capitalize">{connectionStatus}</span>
            </Badge>
          </div>
          
          <div className="flex items-center space-x-2">
            <Button variant="outline" size="sm" onClick={toggleDarkMode}>
              {darkMode ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
            </Button>
            <Button variant="outline" size="sm" onClick={exportConversation} disabled={messages.length === 0}>
              <Download className="w-4 h-4" />
            </Button>
            <Button variant="outline" size="sm" onClick={clearConversation}>
              <Trash2 className="w-4 h-4" />
            </Button>
          </div>
        </div>
        
        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Chat Area */}
          <div className="lg:col-span-3">
            <Tabs value={activeTab} onValueChange={setActiveTab}>
              <TabsList className="grid w-full grid-cols-3">
                <TabsTrigger value="chat">Chat</TabsTrigger>
                <TabsTrigger value="browser">Browser</TabsTrigger>
                <TabsTrigger value="flows">Flows</TabsTrigger>
              </TabsList>
              
              <TabsContent value="chat" className="space-y-4">
                {/* Messages */}
                <Card className="h-[600px] flex flex-col">
                  <CardHeader className="pb-3">
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-lg">Conversation</CardTitle>
                      <div className="flex items-center space-x-2 text-sm text-muted-foreground">
                        <Clock className="w-4 h-4" />
                        <span>{messages.filter(m => m.role !== 'system').length} messages</span>
                      </div>
                    </div>
                  </CardHeader>
                  
                  <CardContent className="flex-1 overflow-hidden">
                    <ScrollArea className="h-full pr-4">
                      <div className="space-y-4">
                        {messages.map((message) => (
                          <div
                            key={message.id}
                            className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                          >
                            <div
                              className={`max-w-[80%] rounded-lg p-3 ${
                                message.role === 'user'
                                  ? 'bg-primary text-primary-foreground'
                                  : message.role === 'system'
                                  ? `bg-muted text-muted-foreground border ${
                                      message.type === 'error' ? 'border-red-500' : 'border-border'
                                    }`
                                  : 'bg-card border border-border'
                              }`}
                            >
                              <div className="flex items-start space-x-2">
                                <div className="flex-shrink-0 mt-1">
                                  {message.role === 'user' ? (
                                    <User className="w-4 h-4" />
                                  ) : message.role === 'system' ? (
                                    <Settings className="w-4 h-4" />
                                  ) : (
                                    <Bot className="w-4 h-4" />
                                  )}
                                </div>
                                <div className="flex-1 min-w-0">
                                  <div className="whitespace-pre-wrap break-words">
                                    {message.content}
                                    {message.streaming && (
                                      <span className="inline-block w-2 h-4 bg-current animate-pulse ml-1" />
                                    )}
                                  </div>
                                  <div className="flex items-center justify-between mt-2 text-xs opacity-70">
                                    <span>{formatTimestamp(message.timestamp)}</span>
                                    {message.model && (
                                      <div className="flex items-center space-x-2">
                                        {message.tokens && (
                                          <span>{message.tokens} tokens</span>
                                        )}
                                        <Badge variant="secondary" className="text-xs">
                                          {message.model.split('/').pop()}
                                        </Badge>
                                      </div>
                                    )}
                                  </div>
                                </div>
                              </div>
                            </div>
                          </div>
                        ))}
                        {isLoading && !streamingEnabled && (
                          <div className="flex justify-start">
                            <div className="bg-card border border-border rounded-lg p-3">
                              <div className="flex items-center space-x-2">
                                <Loader2 className="w-4 h-4 animate-spin" />
                                <span>Thinking...</span>
                              </div>
                            </div>
                          </div>
                        )}
                        <div ref={messagesEndRef} />
                      </div>
                    </ScrollArea>
                  </CardContent>
                </Card>
                
                {/* Input Area */}
                <Card>
                  <CardContent className="p-4">
                    <div className="space-y-4">
                      {/* Model and Task Selection */}
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div className="space-y-2">
                          <label className="text-sm font-medium">Model</label>
                          <Select value={selectedModel} onValueChange={setSelectedModel}>
                            <SelectTrigger>
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              {MODELS.map((model) => (
                                <SelectItem key={model.id} value={model.id}>
                                  <div className="flex items-center justify-between w-full">
                                    <span>{model.name}</span>
                                    <Badge variant="outline" className="ml-2 text-xs">
                                      {model.provider}
                                    </Badge>
                                  </div>
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        </div>
                        
                        <div className="space-y-2">
                          <label className="text-sm font-medium">Task Type</label>
                          <Select value={selectedTaskType} onValueChange={setSelectedTaskType}>
                            <SelectTrigger>
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              {TASK_TYPES.map((task) => {
                                const Icon = task.icon
                                return (
                                  <SelectItem key={task.id} value={task.id}>
                                    <div className="flex items-center space-x-2">
                                      <Icon className="w-4 h-4" />
                                      <span>{task.name}</span>
                                    </div>
                                  </SelectItem>
                                )
                              })}
                            </SelectContent>
                          </Select>
                        </div>
                      </div>
                      
                      {/* Message Input */}
                      <div className="flex space-x-2">
                        <Textarea
                          ref={inputRef}
                          value={inputMessage}
                          onChange={(e) => setInputMessage(e.target.value)}
                          onKeyDown={handleKeyPress}
                          placeholder="Type your message... (Shift+Enter for new line)"
                          className="flex-1 min-h-[60px] max-h-[200px] resize-none"
                          disabled={isLoading}
                        />
                        <Button
                          onClick={sendMessage}
                          disabled={!inputMessage.trim() || isLoading}
                          size="lg"
                          className="px-6"
                        >
                          {isLoading ? (
                            <Loader2 className="w-4 h-4 animate-spin" />
                          ) : (
                            <Send className="w-4 h-4" />
                          )}
                        </Button>
                      </div>
                      
                      {/* Options */}
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-4">
                          <div className="flex items-center space-x-2">
                            <Switch
                              checked={streamingEnabled}
                              onCheckedChange={setStreamingEnabled}
                              id="streaming"
                            />
                            <label htmlFor="streaming" className="text-sm">
                              Streaming
                            </label>
                          </div>
                        </div>
                        
                        <div className="flex items-center space-x-4 text-sm text-muted-foreground">
                          <div className="flex items-center space-x-1">
                            <DollarSign className="w-4 h-4" />
                            <span>${stats.cost.toFixed(4)}</span>
                          </div>
                          <div className="flex items-center space-x-1">
                            <Activity className="w-4 h-4" />
                            <span>{stats.tokens} tokens</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>
              
              <TabsContent value="browser">
                <Card>
                  <CardHeader>
                    <CardTitle>Browser Automation</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-center py-8 text-muted-foreground">
                      <Globe className="w-12 h-12 mx-auto mb-4" />
                      <p>Browser automation features coming soon...</p>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>
              
              <TabsContent value="flows">
                <Card>
                  <CardHeader>
                    <CardTitle>Workflow Execution</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-center py-8 text-muted-foreground">
                      <Zap className="w-12 h-12 mx-auto mb-4" />
                      <p>Workflow execution features coming soon...</p>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>
            </Tabs>
          </div>
          
          {/* Sidebar */}
          <div className="space-y-6">
            {/* Conversations */}
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Conversations</CardTitle>
              </CardHeader>
              <CardContent>
                <ScrollArea className="h-[300px]">
                  <div className="space-y-2">
                    {conversations.map((conv) => (
                      <div
                        key={conv.conversation_id}
                        className={`p-3 rounded-lg border cursor-pointer hover:bg-accent transition-colors ${
                          conversationId === conv.conversation_id ? 'bg-accent' : ''
                        }`}
                        onClick={() => loadConversation(conv.conversation_id)}
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex-1 min-w-0">
                            <p className="text-sm font-medium truncate">
                              Conversation {conv.conversation_id.slice(0, 8)}
                            </p>
                            <p className="text-xs text-muted-foreground">
                              {conv.message_count} messages
                            </p>
                          </div>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={(e) => {
                              e.stopPropagation()
                              deleteConversation(conv.conversation_id)
                            }}
                          >
                            <Trash2 className="w-3 h-3" />
                          </Button>
                        </div>
                      </div>
                    ))}
                    {conversations.length === 0 && (
                      <p className="text-sm text-muted-foreground text-center py-4">
                        No conversations yet
                      </p>
                    )}
                  </div>
                </ScrollArea>
              </CardContent>
            </Card>
            
            {/* Stats */}
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Session Stats</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <MessageSquare className="w-4 h-4" />
                      <span className="text-sm">Requests</span>
                    </div>
                    <span className="font-medium">{stats.requests}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <Activity className="w-4 h-4" />
                      <span className="text-sm">Tokens</span>
                    </div>
                    <span className="font-medium">{stats.tokens.toLocaleString()}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <DollarSign className="w-4 h-4" />
                      <span className="text-sm">Cost</span>
                    </div>
                    <span className="font-medium">${stats.cost.toFixed(4)}</span>
                  </div>
                </div>
              </CardContent>
            </Card>
            
            {/* Quick Actions */}
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Quick Actions</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <Button variant="outline" className="w-full justify-start" onClick={loadConversations}>
                    <MessageSquare className="w-4 h-4 mr-2" />
                    Refresh Conversations
                  </Button>
                  <Button variant="outline" className="w-full justify-start" onClick={exportConversation}>
                    <Download className="w-4 h-4 mr-2" />
                    Export Current Chat
                  </Button>
                  <Button variant="outline" className="w-full justify-start" onClick={clearConversation}>
                    <Trash2 className="w-4 h-4 mr-2" />
                    New Conversation
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App

