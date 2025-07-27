import { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card.jsx'
import { Badge } from '@/components/ui/badge.jsx'
import { Button } from '@/components/ui/button.jsx'
import { Input } from '@/components/ui/input.jsx'
import { Label } from '@/components/ui/label.jsx'
import { Switch } from '@/components/ui/switch.jsx'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select.jsx'
import { Textarea } from '@/components/ui/textarea.jsx'
import { ScrollArea } from '@/components/ui/scroll-area.jsx'
import { Separator } from '@/components/ui/separator.jsx'
import { 
  Activity, 
  Webhook, 
  Zap, 
  Settings, 
  CheckCircle, 
  XCircle, 
  Clock,
  AlertCircle,
  Brain,
  Tool,
  TrendingUp,
  Play,
  Pause,
  Trash2
} from 'lucide-react'

const CALLBACK_EVENTS = [
  { id: 'thinking', name: 'Thinking', description: 'AI reasoning and decision-making', icon: Brain },
  { id: 'tool_use', name: 'Tool Use', description: 'Tool execution events', icon: Tool },
  { id: 'tool_result', name: 'Tool Result', description: 'Tool execution results', icon: CheckCircle },
  { id: 'progress', name: 'Progress', description: 'Task progress updates', icon: TrendingUp },
  { id: 'completion', name: 'Completion', description: 'Task completion events', icon: CheckCircle },
  { id: 'error', name: 'Error', description: 'Error events', icon: XCircle },
  { id: 'model_selection', name: 'Model Selection', description: 'AI model selection events', icon: Settings },
  { id: 'streaming_chunk', name: 'Streaming', description: 'Real-time streaming chunks', icon: Activity }
]

const DELIVERY_METHODS = [
  { id: 'websocket', name: 'WebSocket', description: 'Real-time WebSocket connection' },
  { id: 'webhook', name: 'Webhook', description: 'HTTP POST to webhook URL' },
  { id: 'sse', name: 'Server-Sent Events', description: 'Server-sent events stream' },
  { id: 'polling', name: 'Polling', description: 'Poll for events manually' }
]

export function CallbackPanel({ 
  callbacksEnabled, 
  setCallbacksEnabled, 
  callbackConfig, 
  setCallbackConfig,
  callbackSession,
  setCallbackSession,
  callbackEvents,
  setCallbackEvents,
  callbackStats,
  setCallbackStats,
  apiBaseUrl 
}) {
  const [isCreatingSession, setIsCreatingSession] = useState(false)
  const [isDeletingSession, setIsDeletingSession] = useState(false)
  const [isTestingCallback, setIsTestingCallback] = useState(false)
  const [eventFilter, setEventFilter] = useState('all')

  // Create callback session
  const createCallbackSession = async () => {
    setIsCreatingSession(true)
    try {
      const response = await fetch(`${apiBaseUrl}/callbacks/sessions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          delivery_method: callbackConfig.delivery_method,
          webhook_url: callbackConfig.webhook_url || null,
          events: callbackConfig.events,
          include_intermediate_results: callbackConfig.include_intermediate_results,
          timeout: callbackConfig.timeout,
          retry_attempts: 3,
          headers: callbackConfig.delivery_method === 'webhook' ? {
            'Authorization': 'Bearer your-token-here',
            'X-Source': 'OpenManus-Chat'
          } : null
        })
      })

      if (!response.ok) {
        throw new Error(`Failed to create session: ${response.statusText}`)
      }

      const data = await response.json()
      setCallbackSession(data.session_id)
      setCallbacksEnabled(true)
      
      // Start polling for stats
      pollCallbackStats(data.session_id)
      
    } catch (error) {
      console.error('Error creating callback session:', error)
      alert(`Failed to create callback session: ${error.message}`)
    } finally {
      setIsCreatingSession(false)
    }
  }

  // Delete callback session
  const deleteCallbackSession = async () => {
    if (!callbackSession) return
    
    setIsDeletingSession(true)
    try {
      const response = await fetch(`${apiBaseUrl}/callbacks/sessions/${callbackSession}`, {
        method: 'DELETE'
      })

      if (!response.ok) {
        throw new Error(`Failed to delete session: ${response.statusText}`)
      }

      setCallbackSession(null)
      setCallbacksEnabled(false)
      setCallbackEvents([])
      setCallbackStats(null)
      
    } catch (error) {
      console.error('Error deleting callback session:', error)
      alert(`Failed to delete callback session: ${error.message}`)
    } finally {
      setIsDeletingSession(false)
    }
  }

  // Test callback
  const testCallback = async () => {
    if (!callbackSession) return
    
    setIsTestingCallback(true)
    try {
      const response = await fetch(`${apiBaseUrl}/callbacks/sessions/${callbackSession}/test`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          event_type: 'progress',
          test_data: {
            message: 'This is a test callback from the chat interface',
            progress: 0.5,
            timestamp: new Date().toISOString()
          }
        })
      })

      if (!response.ok) {
        throw new Error(`Failed to send test callback: ${response.statusText}`)
      }

      const data = await response.json()
      alert('Test callback sent successfully!')
      
    } catch (error) {
      console.error('Error testing callback:', error)
      alert(`Failed to send test callback: ${error.message}`)
    } finally {
      setIsTestingCallback(false)
    }
  }

  // Poll callback stats
  const pollCallbackStats = async (sessionId) => {
    try {
      const response = await fetch(`${apiBaseUrl}/callbacks/sessions/${sessionId}/stats`)
      if (response.ok) {
        const stats = await response.json()
        setCallbackStats(stats)
      }
    } catch (error) {
      console.error('Error polling callback stats:', error)
    }
  }

  // Poll for events if using polling delivery method
  const pollEvents = async () => {
    if (!callbackSession || callbackConfig.delivery_method !== 'polling') return
    
    try {
      const response = await fetch(`${apiBaseUrl}/callbacks/sessions/${callbackSession}/events?limit=50`)
      if (response.ok) {
        const data = await response.json()
        setCallbackEvents(prev => [...prev, ...data.events])
      }
    } catch (error) {
      console.error('Error polling events:', error)
    }
  }

  // Update callback config
  const updateCallbackConfig = (key, value) => {
    setCallbackConfig(prev => ({
      ...prev,
      [key]: value
    }))
  }

  // Toggle event in config
  const toggleEvent = (eventId) => {
    setCallbackConfig(prev => ({
      ...prev,
      events: prev.events.includes(eventId)
        ? prev.events.filter(e => e !== eventId)
        : [...prev.events, eventId]
    }))
  }

  // Filter events for display
  const filteredEvents = callbackEvents.filter(event => {
    if (eventFilter === 'all') return true
    return event.event_type === eventFilter
  })

  // Effect for polling
  useEffect(() => {
    if (callbackSession && callbackConfig.delivery_method === 'polling') {
      const interval = setInterval(pollEvents, 5000) // Poll every 5 seconds
      return () => clearInterval(interval)
    }
  }, [callbackSession, callbackConfig.delivery_method])

  // Effect for stats polling
  useEffect(() => {
    if (callbackSession) {
      const interval = setInterval(() => pollCallbackStats(callbackSession), 10000) // Poll every 10 seconds
      return () => clearInterval(interval)
    }
  }, [callbackSession])

  return (
    <div className="space-y-6">
      {/* Callback Configuration */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Webhook className="h-5 w-5" />
            Callback Configuration
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Enable/Disable Callbacks */}
          <div className="flex items-center justify-between">
            <div>
              <Label htmlFor="callbacks-enabled">Enable Callbacks</Label>
              <p className="text-sm text-muted-foreground">
                Receive real-time updates during AI processing
              </p>
            </div>
            <Switch
              id="callbacks-enabled"
              checked={callbacksEnabled}
              onCheckedChange={setCallbacksEnabled}
            />
          </div>

          {callbacksEnabled && (
            <>
              <Separator />
              
              {/* Delivery Method */}
              <div className="space-y-2">
                <Label>Delivery Method</Label>
                <Select
                  value={callbackConfig.delivery_method}
                  onValueChange={(value) => updateCallbackConfig('delivery_method', value)}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {DELIVERY_METHODS.map(method => (
                      <SelectItem key={method.id} value={method.id}>
                        <div>
                          <div className="font-medium">{method.name}</div>
                          <div className="text-sm text-muted-foreground">{method.description}</div>
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {/* Webhook URL (if webhook method) */}
              {callbackConfig.delivery_method === 'webhook' && (
                <div className="space-y-2">
                  <Label htmlFor="webhook-url">Webhook URL</Label>
                  <Input
                    id="webhook-url"
                    placeholder="https://your-webhook-endpoint.com/callback"
                    value={callbackConfig.webhook_url}
                    onChange={(e) => updateCallbackConfig('webhook_url', e.target.value)}
                  />
                </div>
              )}

              {/* Event Selection */}
              <div className="space-y-2">
                <Label>Callback Events</Label>
                <div className="grid grid-cols-2 gap-2">
                  {CALLBACK_EVENTS.map(event => {
                    const Icon = event.icon
                    const isSelected = callbackConfig.events.includes(event.id)
                    
                    return (
                      <div
                        key={event.id}
                        className={`p-3 border rounded-lg cursor-pointer transition-colors ${
                          isSelected 
                            ? 'border-primary bg-primary/10' 
                            : 'border-border hover:border-primary/50'
                        }`}
                        onClick={() => toggleEvent(event.id)}
                      >
                        <div className="flex items-center gap-2">
                          <Icon className="h-4 w-4" />
                          <div>
                            <div className="font-medium text-sm">{event.name}</div>
                            <div className="text-xs text-muted-foreground">{event.description}</div>
                          </div>
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>

              {/* Advanced Options */}
              <div className="space-y-2">
                <Label>Advanced Options</Label>
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="timeout">Timeout (seconds)</Label>
                    <Input
                      id="timeout"
                      type="number"
                      min="5"
                      max="300"
                      value={callbackConfig.timeout}
                      onChange={(e) => updateCallbackConfig('timeout', parseInt(e.target.value))}
                    />
                  </div>
                  <div className="flex items-center space-x-2 pt-6">
                    <Switch
                      id="intermediate-results"
                      checked={callbackConfig.include_intermediate_results}
                      onCheckedChange={(checked) => updateCallbackConfig('include_intermediate_results', checked)}
                    />
                    <Label htmlFor="intermediate-results" className="text-sm">
                      Include intermediate results
                    </Label>
                  </div>
                </div>
              </div>

              {/* Session Management */}
              <Separator />
              <div className="flex gap-2">
                {!callbackSession ? (
                  <Button 
                    onClick={createCallbackSession} 
                    disabled={isCreatingSession}
                    className="flex-1"
                  >
                    {isCreatingSession ? (
                      <>
                        <Activity className="h-4 w-4 mr-2 animate-spin" />
                        Creating Session...
                      </>
                    ) : (
                      <>
                        <Play className="h-4 w-4 mr-2" />
                        Create Callback Session
                      </>
                    )}
                  </Button>
                ) : (
                  <>
                    <Button 
                      onClick={testCallback} 
                      disabled={isTestingCallback}
                      variant="outline"
                    >
                      {isTestingCallback ? (
                        <>
                          <Activity className="h-4 w-4 mr-2 animate-spin" />
                          Testing...
                        </>
                      ) : (
                        <>
                          <Zap className="h-4 w-4 mr-2" />
                          Test
                        </>
                      )}
                    </Button>
                    <Button 
                      onClick={deleteCallbackSession} 
                      disabled={isDeletingSession}
                      variant="destructive"
                      className="flex-1"
                    >
                      {isDeletingSession ? (
                        <>
                          <Activity className="h-4 w-4 mr-2 animate-spin" />
                          Deleting...
                        </>
                      ) : (
                        <>
                          <Trash2 className="h-4 w-4 mr-2" />
                          Delete Session
                        </>
                      )}
                    </Button>
                  </>
                )}
              </div>
            </>
          )}
        </CardContent>
      </Card>

      {/* Session Status */}
      {callbackSession && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5" />
              Session Status
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Session ID:</span>
                <Badge variant="outline" className="font-mono text-xs">
                  {callbackSession.slice(0, 8)}...
                </Badge>
              </div>
              
              {callbackStats && (
                <div className="grid grid-cols-2 gap-4">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-green-600">{callbackStats.events_delivered}</div>
                    <div className="text-sm text-muted-foreground">Delivered</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-red-600">{callbackStats.events_failed}</div>
                    <div className="text-sm text-muted-foreground">Failed</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold">{callbackStats.queued_events}</div>
                    <div className="text-sm text-muted-foreground">Queued</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-blue-600">
                      {Math.round(callbackStats.delivery_rate * 100)}%
                    </div>
                    <div className="text-sm text-muted-foreground">Success Rate</div>
                  </div>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Callback Events */}
      {callbackEvents.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Clock className="h-5 w-5" />
              Callback Events
            </CardTitle>
            <div className="flex gap-2">
              <Select value={eventFilter} onValueChange={setEventFilter}>
                <SelectTrigger className="w-40">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Events</SelectItem>
                  {CALLBACK_EVENTS.map(event => (
                    <SelectItem key={event.id} value={event.id}>
                      {event.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <Button 
                variant="outline" 
                size="sm"
                onClick={() => setCallbackEvents([])}
              >
                <Trash2 className="h-4 w-4 mr-2" />
                Clear
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            <ScrollArea className="h-64">
              <div className="space-y-2">
                {filteredEvents.map((event, index) => {
                  const eventConfig = CALLBACK_EVENTS.find(e => e.id === event.event_type)
                  const Icon = eventConfig?.icon || AlertCircle
                  
                  return (
                    <div key={index} className="p-3 border rounded-lg">
                      <div className="flex items-start gap-3">
                        <Icon className="h-4 w-4 mt-1 text-muted-foreground" />
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2">
                            <Badge variant="outline" className="text-xs">
                              {event.event_type}
                            </Badge>
                            <span className="text-xs text-muted-foreground">
                              {new Date(event.timestamp).toLocaleTimeString()}
                            </span>
                          </div>
                          <div className="mt-1 text-sm">
                            <pre className="whitespace-pre-wrap text-xs bg-muted p-2 rounded mt-2">
                              {JSON.stringify(event.data, null, 2)}
                            </pre>
                          </div>
                        </div>
                      </div>
                    </div>
                  )
                })}
              </div>
            </ScrollArea>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

