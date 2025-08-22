// Hedge Fund Dashboard Theme
// Centralized theme configuration for consistent styling

export const hedgeFundTheme = {
  colors: {
    // Background colors
    background: {
      primary: '#0a0e1a',
      secondary: '#1e293b',
      card: '#0f1419',
      success: '#0f1a0f',
      error: '#1a0f0f',
      warning: '#1a1a0f',
    },
    // Border colors
    border: {
      primary: '#334155',
      secondary: '#374151',
      success: '#10b981',
      error: '#ef4444',
      warning: '#f59e0b',
      info: '#3b82f6',
    },
    // Text colors
    text: {
      primary: '#f1f5f9',
      secondary: '#94a3b8',
      disabled: '#64748b',
      success: '#10b981',
      error: '#ef4444',
      warning: '#f59e0b',
      info: '#3b82f6',
    },
  },
  spacing: {
    xs: '0.25rem',
    sm: '0.5rem',
    md: '1rem',
    lg: '1.5rem',
    xl: '2rem',
  },
  borderRadius: {
    sm: '0.25rem',
    md: '0.5rem',
    lg: '1rem',
  },
};

// Common styles
export const commonStyles = {
  card: {
    backgroundColor: hedgeFundTheme.colors.background.card,
    border: `1px solid ${hedgeFundTheme.colors.border.primary}`,
    borderRadius: hedgeFundTheme.borderRadius.md,
    height: '100%',
  },
  table: {
    backgroundColor: hedgeFundTheme.colors.background.secondary,
    border: `1px solid ${hedgeFundTheme.colors.border.primary}`,
    '& .MuiTableCell-root': {
      borderColor: hedgeFundTheme.colors.border.secondary,
    },
  },
  tableHeader: {
    color: hedgeFundTheme.colors.text.secondary,
    fontWeight: 600,
  },
  tableCell: {
    color: hedgeFundTheme.colors.text.primary,
  },
};

export default hedgeFundTheme;
