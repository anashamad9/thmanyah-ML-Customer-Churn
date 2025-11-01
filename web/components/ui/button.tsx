"use client";

import * as React from "react";
import { cn } from "../../lib/utils";

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "solid" | "outline" | "ghost";
  size?: "default" | "sm";
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = "solid", size = "default", type, ...props }, ref) => {
    const variants: Record<ButtonProps["variant"], string> = {
      solid: "btn-solid",
      outline: "btn-outline",
      ghost: "btn-ghost",
    };
    const sizes: Record<ButtonProps["size"], string> = {
      default: "",
      sm: "btn-sm",
    };

    return (
      <button
        ref={ref}
        type={type ?? "button"}
        className={cn("btn", variants[variant], sizes[size], className)}
        {...props}
      />
    );
  },
);

Button.displayName = "Button";
