# Physics-Informed Neural Network for Local Volatility Option Pricing

This project develops a **Physics-Informed Neural Network (PINN)** to approximate solutions of the **Black–Scholes partial differential equation with local volatility** for European option pricing.

Training data are generated using a **Crank–Nicolson finite difference solver** under a parametric local volatility model. The PINN learns option price surfaces while enforcing the governing PDE, boundary conditions, and terminal payoff constraints through a composite loss function. Model performance is compared against a baseline neural network.

## Repository Structure

