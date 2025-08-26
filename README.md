# High-Performance Parallel N-Body Simulation

Real-Time Gravitational Dynamics Simulation with Pygame UI, Python, PyCUDA, and MPI

📌 **Project Overview**

This project explores high-performance computing (HPC) techniques for simulating the gravitational interactions of multiple bodies in 2D/3D space.  
We implement and benchmark three approaches:

- **Serial Python Implementation**
    
- **MPI Parallelism (`mpi4py`)**
    
- **GPU Acceleration (PyCUDA)**
    

The simulation includes a **real-time UI using Pygame**, which runs on a **separate thread** to allow smooth visualization while computations (forces, integration, and updates) run concurrently in the main simulation thread.  
The study aims to analyse **speedup**, **scalability**, and **accuracy** across these methods for real-time simulation of planetary systems, star clusters, and particle dynamics.

---

👨‍💻 **Team Members**

- **BATHINA HARSHINA REDDY** - CB.AI.U4AID23108
    
- **M. SRINIDH** - CB.AI.U4AID23124
    
- **PRATHAP P** - CB.AI.U4AID23160
    
- **ARUNMANIKANDAN C** - CB.AI.U4AID23167
    

---

⚙️ **Features**

- Real-time N-body gravitational simulation with configurable number of bodies
    
- Multiple numerical integration methods (Euler, Verlet, Runge-Kutta optional)
    
- **Serial, MPI, and GPU (PyCUDA) implementations**
    
- **Pygame-based UI** running in a **separate thread** for smooth visualisation
    
- Modular and extendable code structure
    
- Cross-platform support: **Windows** and **Linux**
    
