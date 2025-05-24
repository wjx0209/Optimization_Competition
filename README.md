# 🧠 Optimization_Competition

This repository presents an optimization method developed by SJTU to solve the **point heat conduction problem** for competition purposes.

---

## 📌 Problem Overview

The **volume-to-point heat conduction problem** is one of the fundamental challenges in the **thermal management of electronic devices**.

The domain contains:
- An **internal heat source**
- Boundaries with **fixed temperatures** or **heat flows**

---

## 🎯 Task Objective

> **Find the optimal continuous high thermal conductivity material distribution**,  
> under a fixed volume fraction constraint (15%),  
> to **minimize the average temperature** in the domain.

---

## 📐 Problem Schematic

<p align="center">
  <img src="images/VP_diagram.png" alt="VP diagram" width="400">
</p>

**Fig. 1**. The schematic diagram of the volume-to-point (VP) heat conduction problem.  
The region contains a high conductivity filling material and a uniform heat source.

---

## ⚙️ Algorithm Description

This code uses a **greedy algorithm** to solve the VP problem and imposes a **connectivity constraint** to ensure that the high-conductivity region remains physically reasonable.

---

## 📁 Directory Structure

```text
.
├── images/
│   └── VP_diagram.png         # Schematic image of the problem
├── main.py                    # Main script with greedy algorithm
├── README.md                  # This file


