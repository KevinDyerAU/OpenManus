# OpenManus UI Components Documentation

## Overview

This directory contains all the UI components for the OpenManus AI Agent Platform web interface. The components are built using React 18+ with modern hooks and follow best practices for maintainability and reusability.

## Component Architecture

### Core Components

#### `ErrorBoundary.jsx`
**Purpose**: Provides application-wide error handling and graceful error recovery.

**Props**:
- `children` (ReactNode): Child components to wrap with error boundary

**Features**:
- Catches JavaScript errors in component tree
- Displays user-friendly error messages
- Shows detailed error information in development mode
- Provides retry and refresh options
- Logs errors for debugging

**Usage**:
```jsx
import ErrorBoundary from '@/components/ErrorBoundary.jsx'

<ErrorBoundary>
  <YourComponent />
</ErrorBoundary>
```

#### `LoadingState.jsx`
**Purpose**: Provides consistent loading states throughout the application.

**Components**:
- `LoadingState`: Main loading component with multiple variants
- `NetworkErrorState`: Specialized component for network error handling

**LoadingState Props**:
- `message` (string): Loading message to display (default: "Loading...")
- `variant` (string): Loading variant - 'default', 'minimal', 'chat', 'fullscreen'
- `showIcon` (boolean): Whether to show loading icon (default: true)
- `className` (string): Additional CSS classes

**NetworkErrorState Props**:
- `onRetry` (function): Callback function for retry action
- `message` (string): Error message to display

**Usage**:
```jsx
import { LoadingState, NetworkErrorState } from '@/components/LoadingState.jsx'

// Basic loading state
<LoadingState message="Loading data..." />

// Chat typing indicator
<LoadingState variant="chat" />

// Network error with retry
<NetworkErrorState 
  message="Failed to connect to server" 
  onRetry={() => window.location.reload()} 
/>
```

### UI Components (`/ui` directory)

These components are based on Radix UI primitives and styled with Tailwind CSS, following the shadcn/ui design system.

#### `button.jsx`
**Purpose**: Reusable button component with multiple variants and sizes.

**Props**:
- `variant`: 'default', 'destructive', 'outline', 'secondary', 'ghost', 'link'
- `size`: 'default', 'sm', 'lg', 'icon'
- `asChild` (boolean): Render as child component
- Standard HTML button attributes

#### `card.jsx`
**Purpose**: Container component for grouping related content.

**Components**:
- `Card`: Main container
- `CardHeader`: Header section
- `CardTitle`: Title component
- `CardDescription`: Description component
- `CardContent`: Main content area
- `CardFooter`: Footer section

#### `input.jsx`
**Purpose**: Text input component with consistent styling.

**Props**:
- `type`: Input type (text, email, password, etc.)
- Standard HTML input attributes

#### `select.jsx`
**Purpose**: Dropdown selection component.

**Components**:
- `Select`: Main select container
- `SelectTrigger`: Clickable trigger
- `SelectValue`: Display selected value
- `SelectContent`: Dropdown content
- `SelectItem`: Individual option

#### `tabs.jsx`
**Purpose**: Tab navigation component.

**Components**:
- `Tabs`: Main container
- `TabsList`: Tab navigation list
- `TabsTrigger`: Individual tab trigger
- `TabsContent`: Tab content panel

#### `switch.jsx`
**Purpose**: Toggle switch component.

**Props**:
- `checked` (boolean): Switch state
- `onCheckedChange` (function): Change handler

#### `label.jsx`
**Purpose**: Form label component with proper accessibility.

**Props**:
- Standard HTML label attributes
- Automatically associates with form controls

#### `textarea.jsx`
**Purpose**: Multi-line text input component.

**Props**:
- Standard HTML textarea attributes

#### `scroll-area.jsx`
**Purpose**: Custom scrollable area with styled scrollbars.

**Props**:
- `className`: Additional CSS classes
- Standard div attributes

#### `separator.jsx`
**Purpose**: Visual separator line component.

**Props**:
- `orientation`: 'horizontal' or 'vertical'
- `decorative` (boolean): Whether separator is decorative

#### `badge.jsx`
**Purpose**: Small status or category indicator.

**Props**:
- `variant`: 'default', 'secondary', 'destructive', 'outline'

## Styling and Theming

### Design System
- **Colors**: Uses CSS custom properties for theme support
- **Typography**: Tailwind CSS typography classes
- **Spacing**: Consistent spacing scale using Tailwind
- **Dark Mode**: Automatic dark/light theme support

### CSS Classes
- All components use Tailwind CSS for styling
- Custom CSS in `App.css` for global styles and animations
- CSS variables for theme colors in `:root` and `[data-theme="dark"]`

## State Management

### Local State
- Components use `useState` for local state management
- `useEffect` for side effects and lifecycle management
- `useRef` for DOM references and mutable values

### Global State
- Application state managed in main `App.jsx` component
- Props drilling for state sharing between components
- Context API can be added for more complex state needs

## API Integration

### HTTP Requests
- Uses native `fetch` API for HTTP requests
- Error handling with try/catch blocks
- Loading states during API calls

### WebSocket Communication
- Real-time communication via WebSocket
- Automatic reconnection handling
- Connection status monitoring

### Configuration
- Environment variables via Vite (`import.meta.env`)
- `VITE_API_URL`: Backend API base URL
- `VITE_WS_URL`: WebSocket connection URL

## Testing

### Unit Tests
- Located in `/src/test/components/`
- Uses Vitest and React Testing Library
- Tests component rendering, user interactions, and props

### End-to-End Tests
- Located in `/tests/e2e/`
- Uses Playwright for browser automation
- Tests complete user workflows and responsive design

### Running Tests
```bash
# Unit tests
npm run test

# E2E tests
npm run test:e2e

# Test coverage
npm run test:coverage
```

## Best Practices

### Component Design
1. **Single Responsibility**: Each component has a clear, focused purpose
2. **Reusability**: Components are designed to be reused across the application
3. **Props Interface**: Clear and well-documented props with TypeScript-style comments
4. **Error Handling**: Graceful error handling and fallback states

### Performance
1. **React.memo**: Use for expensive components that re-render frequently
2. **useCallback**: Memoize event handlers passed to child components
3. **useMemo**: Memoize expensive calculations
4. **Code Splitting**: Use dynamic imports for large components

### Accessibility
1. **Semantic HTML**: Use proper HTML elements and ARIA attributes
2. **Keyboard Navigation**: Ensure all interactive elements are keyboard accessible
3. **Screen Readers**: Provide proper labels and descriptions
4. **Color Contrast**: Maintain sufficient color contrast ratios

### Code Quality
1. **ESLint**: Follow configured linting rules
2. **Consistent Naming**: Use descriptive and consistent naming conventions
3. **Documentation**: Document complex logic and component APIs
4. **Error Boundaries**: Wrap components that might throw errors

## Contributing

When adding new components:

1. **Location**: Place in appropriate directory (`/components` or `/components/ui`)
2. **Documentation**: Add JSDoc comments for props and functionality
3. **Testing**: Include unit tests for new components
4. **Styling**: Follow existing design system and Tailwind conventions
5. **Accessibility**: Ensure components meet accessibility standards
6. **Error Handling**: Include appropriate error states and boundaries
