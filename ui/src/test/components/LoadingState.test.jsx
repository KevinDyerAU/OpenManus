import { render, screen } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'
import userEvent from '@testing-library/user-event'
import { LoadingState, NetworkErrorState } from '../../components/LoadingState.jsx'

describe('LoadingState Component', () => {
  it('renders default loading state with message', () => {
    render(<LoadingState message="Loading data..." />)
    expect(screen.getByText('Loading data...')).toBeInTheDocument()
  })

  it('renders minimal variant without icon when showIcon is false', () => {
    render(<LoadingState variant="minimal" showIcon={false} message="Processing..." />)
    expect(screen.getByText('Processing...')).toBeInTheDocument()
    expect(screen.queryByRole('img')).not.toBeInTheDocument()
  })

  it('renders chat variant with typing animation', () => {
    render(<LoadingState variant="chat" />)
    const dots = screen.container.querySelectorAll('.animate-bounce')
    expect(dots).toHaveLength(3)
  })

  it('renders fullscreen variant', () => {
    render(<LoadingState variant="fullscreen" message="Initializing..." />)
    expect(screen.getByText('Initializing...')).toBeInTheDocument()
  })
})

describe('NetworkErrorState Component', () => {
  it('renders error message', () => {
    render(<NetworkErrorState message="Connection failed" />)
    expect(screen.getByText('Connection failed')).toBeInTheDocument()
  })

  it('calls onRetry when retry button is clicked', async () => {
    const user = userEvent.setup()
    const mockRetry = vi.fn()
    
    render(<NetworkErrorState onRetry={mockRetry} />)
    
    const retryButton = screen.getByText('Retry')
    await user.click(retryButton)
    
    expect(mockRetry).toHaveBeenCalledTimes(1)
  })

  it('does not render retry button when onRetry is not provided', () => {
    render(<NetworkErrorState />)
    expect(screen.queryByText('Retry')).not.toBeInTheDocument()
  })
})
