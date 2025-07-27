import { test, expect } from '@playwright/test'

test.describe('OpenManus UI', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
  })

  test('should load the main application', async ({ page }) => {
    // Check that the main title is visible
    await expect(page.locator('h1')).toContainText('OpenManus')
  })

  test('should navigate between tabs', async ({ page }) => {
    // Test tab navigation
    await page.click('[data-testid="settings-tab"]')
    await expect(page.locator('[data-testid="settings-panel"]')).toBeVisible()
    
    await page.click('[data-testid="chat-tab"]')
    await expect(page.locator('[data-testid="chat-panel"]')).toBeVisible()
  })

  test('should handle model selection', async ({ page }) => {
    // Navigate to settings
    await page.click('[data-testid="settings-tab"]')
    
    // Open model selector
    await page.click('[data-testid="model-selector"]')
    
    // Select a model
    await page.click('text=GPT-4o')
    
    // Verify selection
    await expect(page.locator('[data-testid="model-selector"]')).toContainText('GPT-4o')
  })

  test('should display connection status', async ({ page }) => {
    // Check for connection status indicator
    const statusIndicator = page.locator('[data-testid="connection-status"]')
    await expect(statusIndicator).toBeVisible()
  })

  test('should be responsive on mobile', async ({ page }) => {
    // Set mobile viewport
    await page.setViewportSize({ width: 375, height: 667 })
    
    // Check that the app is still functional
    await expect(page.locator('h1')).toBeVisible()
    
    // Check that navigation is accessible
    const tabs = page.locator('[role="tablist"]')
    await expect(tabs).toBeVisible()
  })
})
