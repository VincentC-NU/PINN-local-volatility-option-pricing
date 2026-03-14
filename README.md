# Physics-Informed Neural Network for Local Volatility Option Pricing

This project develops a **Physics-Informed Neural Network (PINN)** to approximate solutions of the **Black–Scholes partial differential equation with local volatility** for European option pricing.

Training data are generated using a **Crank–Nicolson finite difference solver** under a parametric local volatility model. The PINN learns option price surfaces while enforcing the governing PDE, boundary conditions, and terminal payoff constraints through a composite loss function. Model performance is compared against a baseline neural network.

## Overview

This project develops a Physics-Informed Neural Network (PINN) to approximate solutions of the Black–Scholes partial differential equation with a parametric local volatility model for European option pricing.

Traditional neural networks trained only on data may violate the governing financial dynamics. A PINN incorporates the Black–Scholes PDE directly into the training objective, enforcing physical and financial constraints while learning option price surfaces.

Training data are generated using a Crank–Nicolson finite difference solver across randomly sampled volatility surfaces.

The model is evaluated against a baseline multilayer perceptron (MLP). Results show that the PINN significantly improves PDE consistency while maintaining comparable pricing accuracy.

## Black–Scholes PDE with Constant Volatility
Under the classical Black–Scholes model, the volatility σ is assumed to be constant. The option price V(S,t) satisfies:

$$ \frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + rS \frac{\partial V}{\partial S} - rV = 0 $$

with terminal condition

$$
V(S,T)=\max(S-K,0)
$$

and boundary conditions

$$
V(0,t)=0
$$

$$
V(S,t)\to S-Ke^{-r(T-t)} \quad \text{as } S\to\infty
$$

In the constant-volatility case, the equation has a closed-form analytical solution for call and put options.

Call option price:

$$
C(S_t, t)=S_0N(d_1)-Ke^{-r(T-t)}N(d_2)
$$

Put option price:

$$
P(S_t, t)=Ke^{-r(T-t)}N(-d_2)-S_0N(-d_1)
$$

This analytical solution provides an important benchmark for numerical methods and machine learning models.

## Black–Scholes PDE with Local Volatility
However, empirical observations of financial markets show that volatility is not constant but varies with both the underlying asset price and time. This produces well-known phenomena such as the **volatility smile** and motivates the use of **local volatility models**, where the volatility parameter becomes a function σ(S,t). As a result, the Black-Scholes PDE becomes:

$$ \frac{\partial V}{\partial t} + \frac{1}{2}\sigma(S,t)^2 S^2 \frac{\partial^2 V}{\partial S^2} + rS \frac{\partial V}{\partial S} - rV = 0 $$

Unlike the constant-volatility case, the local-volatility Black–Scholes equation generally does not admit a closed-form analytical solution. As a result, numerical methods or machine learning approaches must be used to approximate the option price surface.



