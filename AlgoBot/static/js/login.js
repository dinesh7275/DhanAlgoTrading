/* Login JavaScript for Dhan Algorithmic Trading */

class LoginManager {
    constructor() {
        this.form = null;
        this.isLoading = false;
        this.init();
    }

    init() {
        this.setupForm();
        this.setupEventListeners();
        this.setupValidation();
        console.log('Login manager initialized');
    }

    setupForm() {
        this.form = document.getElementById('login-form');
        if (!this.form) {
            console.error('Login form not found');
            return;
        }
    }

    setupEventListeners() {
        // Form submission
        if (this.form) {
            this.form.addEventListener('submit', (e) => this.handleLogin(e));
        }

        // Password visibility toggle
        const passwordToggle = document.querySelector('.password-toggle');
        if (passwordToggle) {
            passwordToggle.addEventListener('click', () => this.togglePasswordVisibility());
        }

        // Demo login button
        const demoBtn = document.getElementById('demo-login');
        if (demoBtn) {
            demoBtn.addEventListener('click', () => this.handleDemoLogin());
        }

        // Input field animations
        document.querySelectorAll('.form-input').forEach(input => {
            input.addEventListener('focus', (e) => this.handleInputFocus(e));
            input.addEventListener('blur', (e) => this.handleInputBlur(e));
            input.addEventListener('input', (e) => this.handleInputChange(e));
        });

        // Remember me functionality
        const rememberCheckbox = document.getElementById('remember-me');
        if (rememberCheckbox) {
            rememberCheckbox.addEventListener('change', (e) => this.handleRememberMe(e));
        }

        // Enter key handling
        document.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && this.form && !this.isLoading) {
                this.form.dispatchEvent(new Event('submit'));
            }
        });
    }

    setupValidation() {
        // Real-time validation
        const emailInput = document.getElementById('email');
        const passwordInput = document.getElementById('password');

        if (emailInput) {
            emailInput.addEventListener('input', () => this.validateEmail(emailInput));
        }

        if (passwordInput) {
            passwordInput.addEventListener('input', () => this.validatePassword(passwordInput));
        }
    }

    async handleLogin(event) {
        event.preventDefault();
        
        if (this.isLoading) return;

        const formData = new FormData(this.form);
        const credentials = {
            email: formData.get('email'),
            password: formData.get('password'),
            remember_me: formData.get('remember_me') === 'on'
        };

        // Validate inputs
        if (!this.validateForm(credentials)) {
            return;
        }

        this.setLoading(true);

        try {
            const response = await fetch('/api/auth/login', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Requested-With': 'XMLHttpRequest'
                },
                body: JSON.stringify(credentials)
            });

            const data = await response.json();

            if (data.success) {
                this.showMessage('Login successful! Redirecting...', 'success');
                
                // Store auth token if provided
                if (data.token) {
                    localStorage.setItem('auth_token', data.token);
                }

                // Store user preferences
                if (credentials.remember_me) {
                    localStorage.setItem('remember_email', credentials.email);
                } else {
                    localStorage.removeItem('remember_email');
                }

                // Redirect to dashboard
                setTimeout(() => {
                    window.location.href = data.redirect_url || '/dashboard';
                }, 1500);

            } else {
                throw new Error(data.message || 'Login failed');
            }

        } catch (error) {
            console.error('Login error:', error);
            this.showMessage(error.message || 'Login failed. Please try again.', 'error');
        } finally {
            this.setLoading(false);
        }
    }

    async handleDemoLogin() {
        if (this.isLoading) return;

        this.setLoading(true);

        try {
            const response = await fetch('/api/auth/demo-login', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Requested-With': 'XMLHttpRequest'
                }
            });

            const data = await response.json();

            if (data.success) {
                this.showMessage('Demo login successful! Redirecting...', 'success');
                
                setTimeout(() => {
                    window.location.href = data.redirect_url || '/dashboard';
                }, 1500);

            } else {
                throw new Error(data.message || 'Demo login failed');
            }

        } catch (error) {
            console.error('Demo login error:', error);
            this.showMessage(error.message || 'Demo login failed. Please try again.', 'error');
        } finally {
            this.setLoading(false);
        }
    }

    validateForm(credentials) {
        let isValid = true;

        // Validate email
        if (!credentials.email) {
            this.showFieldError('email', 'Email is required');
            isValid = false;
        } else if (!this.isValidEmail(credentials.email)) {
            this.showFieldError('email', 'Please enter a valid email address');
            isValid = false;
        } else {
            this.clearFieldError('email');
        }

        // Validate password
        if (!credentials.password) {
            this.showFieldError('password', 'Password is required');
            isValid = false;
        } else if (credentials.password.length < 6) {
            this.showFieldError('password', 'Password must be at least 6 characters');
            isValid = false;
        } else {
            this.clearFieldError('password');
        }

        return isValid;
    }

    validateEmail(input) {
        const email = input.value.trim();
        
        if (!email) {
            input.setCustomValidity('Email is required');
            return false;
        }
        
        if (!this.isValidEmail(email)) {
            input.setCustomValidity('Please enter a valid email address');
            return false;
        }
        
        input.setCustomValidity('');
        return true;
    }

    validatePassword(input) {
        const password = input.value;
        
        if (!password) {
            input.setCustomValidity('Password is required');
            return false;
        }
        
        if (password.length < 6) {
            input.setCustomValidity('Password must be at least 6 characters');
            return false;
        }
        
        input.setCustomValidity('');
        return true;
    }

    isValidEmail(email) {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return emailRegex.test(email);
    }

    showFieldError(fieldName, message) {
        const field = document.getElementById(fieldName);
        if (!field) return;

        // Remove existing error
        this.clearFieldError(fieldName);

        // Add error class
        field.classList.add('error');

        // Create error message
        const errorDiv = document.createElement('div');
        errorDiv.className = 'field-error';
        errorDiv.textContent = message;
        errorDiv.id = `${fieldName}-error`;

        // Insert after the field
        field.parentNode.insertBefore(errorDiv, field.nextSibling);
    }

    clearFieldError(fieldName) {
        const field = document.getElementById(fieldName);
        if (field) {
            field.classList.remove('error');
        }

        const errorDiv = document.getElementById(`${fieldName}-error`);
        if (errorDiv) {
            errorDiv.remove();
        }
    }

    togglePasswordVisibility() {
        const passwordInput = document.getElementById('password');
        const toggleBtn = document.querySelector('.password-toggle');
        
        if (!passwordInput || !toggleBtn) return;

        const isPassword = passwordInput.type === 'password';
        passwordInput.type = isPassword ? 'text' : 'password';
        toggleBtn.textContent = isPassword ? 'Hide' : 'Show';
    }

    handleInputFocus(event) {
        const formGroup = event.target.closest('.form-group');
        if (formGroup) {
            formGroup.classList.add('focused');
        }
    }

    handleInputBlur(event) {
        const formGroup = event.target.closest('.form-group');
        if (formGroup) {
            formGroup.classList.remove('focused');
        }
    }

    handleInputChange(event) {
        const input = event.target;
        const formGroup = input.closest('.form-group');
        
        if (formGroup) {
            if (input.value.trim()) {
                formGroup.classList.add('has-value');
            } else {
                formGroup.classList.remove('has-value');
            }
        }

        // Clear field errors on input change
        this.clearFieldError(input.id);
    }

    handleRememberMe(event) {
        const isChecked = event.target.checked;
        
        if (isChecked) {
            // Show info about remember me
            this.showMessage('Your email will be remembered for next login', 'info', 3000);
        }
    }

    setLoading(loading) {
        this.isLoading = loading;
        const submitBtn = document.querySelector('.btn-login');
        const demoBtn = document.getElementById('demo-login');
        
        if (submitBtn) {
            submitBtn.disabled = loading;
            if (loading) {
                submitBtn.innerHTML = '<span class="loading-spinner"></span> Signing in...';
            } else {
                submitBtn.innerHTML = 'Sign In';
            }
        }

        if (demoBtn) {
            demoBtn.disabled = loading;
        }

        // Disable form inputs
        if (this.form) {
            const inputs = this.form.querySelectorAll('input');
            inputs.forEach(input => {
                input.disabled = loading;
            });
        }
    }

    showMessage(message, type = 'info', duration = 5000) {
        // Remove existing messages
        document.querySelectorAll('.alert').forEach(alert => alert.remove());

        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type}`;
        alertDiv.textContent = message;

        // Insert at the top of the form
        if (this.form) {
            this.form.insertBefore(alertDiv, this.form.firstChild);
        }

        // Auto-remove after duration
        if (duration > 0) {
            setTimeout(() => {
                alertDiv.remove();
            }, duration);
        }
    }

    // Load remembered email on page load
    loadRememberedCredentials() {
        const rememberedEmail = localStorage.getItem('remember_email');
        if (rememberedEmail) {
            const emailInput = document.getElementById('email');
            const rememberCheckbox = document.getElementById('remember-me');
            
            if (emailInput) {
                emailInput.value = rememberedEmail;
                this.handleInputChange({ target: emailInput });
            }
            
            if (rememberCheckbox) {
                rememberCheckbox.checked = true;
            }
        }
    }

    // Check if user is already logged in
    async checkAuthStatus() {
        const token = localStorage.getItem('auth_token');
        if (token) {
            try {
                const response = await fetch('/api/auth/verify', {
                    method: 'GET',
                    headers: {
                        'Authorization': `Bearer ${token}`,
                        'X-Requested-With': 'XMLHttpRequest'
                    }
                });

                if (response.ok) {
                    const data = await response.json();
                    if (data.valid) {
                        // User is already logged in, redirect to dashboard
                        window.location.href = '/dashboard';
                        return;
                    }
                }
            } catch (error) {
                console.log('Auth check failed:', error);
            }
            
            // Remove invalid token
            localStorage.removeItem('auth_token');
        }
    }
}

// Initialize login manager when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const loginManager = new LoginManager();
    
    // Load remembered credentials
    loginManager.loadRememberedCredentials();
    
    // Check auth status
    loginManager.checkAuthStatus();
    
    // Make globally available
    window.LoginManager = loginManager;
});

// Handle page visibility change to refresh auth status
document.addEventListener('visibilitychange', () => {
    if (!document.hidden && window.LoginManager) {
        window.LoginManager.checkAuthStatus();
    }
});

// Add CSS for field errors
const style = document.createElement('style');
style.textContent = `
    .form-input.error {
        border-color: #dc3545 !important;
        box-shadow: 0 0 0 3px rgba(220, 53, 69, 0.1) !important;
    }
    
    .field-error {
        color: #dc3545;
        font-size: 0.875rem;
        margin-top: 0.25rem;
        display: block;
    }
    
    .form-group.focused .form-label {
        color: #667eea;
    }
    
    .form-group.has-value .form-input {
        background-color: #fff;
    }
`;
document.head.appendChild(style);