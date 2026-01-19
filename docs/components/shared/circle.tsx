"use client";

import { motion } from "motion/react";
import clsx from "clsx";
import { useCallback, useEffect, useRef } from "react";
import { useSetAtom } from "jotai";
import { centersAtom, hoveredNodeAtom } from "./atoms";

export const LAYOUT_ANIMATION_DURATION = 400;

interface CircleProps {
  children: React.ReactNode;
  className?: string;
  index: number;
  value: string;
  type: "input" | "output";
  color: string;
  isMasked?: boolean;
  dimmed?: boolean;
  onClick?: () => void;
}

const Circle = ({
  children,
  className,
  index,
  value,
  type,
  color,
  isMasked = false,
  dimmed = false,
  onClick,
}: CircleProps) => {
  const ref = useRef<HTMLDivElement>(null);
  const setCenters = useSetAtom(centersAtom);
  const setHoveredNode = useSetAtom(hoveredNodeAtom);

  // Create a unique pattern ID for masked circles
  const patternId = `masked-pattern-${index}-${type}`;

  const updateCenter = useCallback(() => {
    if (!ref.current) return;
    const rect = ref.current.getBoundingClientRect();
    setCenters((prevCenters) => ({
      ...prevCenters,
      [value]: {
        x: rect.left + rect.width / 2,
        y: rect.top + rect.height / 2,
      },
    }));
  }, [value, setCenters]);

  // Update center on mount and scroll
  useEffect(() => {
    updateCenter();

    // Update centers when scrolling
    window.addEventListener("scroll", updateCenter, true);
    window.addEventListener("resize", updateCenter);

    return () => {
      window.removeEventListener("scroll", updateCenter, true);
      window.removeEventListener("resize", updateCenter);
    };
  }, [updateCenter]);

  return (
    <motion.div
      ref={ref}
      layout
      onLayoutAnimationComplete={updateCenter}
      onHoverStart={() => setHoveredNode({ index, type })}
      onHoverEnd={() => setHoveredNode(null)}
      onClick={onClick}
      initial={{ scale: 1 }}
      animate={{ scale: 1, opacity: dimmed ? 0.3 : 1 }}
      transition={{
        layout: {
          duration: LAYOUT_ANIMATION_DURATION / 1000,
          ease: "easeInOut",
        },
        scale: { duration: 0.2, ease: "easeInOut" },
        opacity: { duration: 0.2 },
      }}
      whileHover={{
        scale: 1.1,
      }}
      style={{
        backgroundColor: isMasked ? "" : color,
        border: isMasked ? `2px solid ${color}` : undefined,
      }}
      className={clsx(
        [
          "w-12 h-12 select-none font-heading rounded-full flex items-center justify-center",
          "relative overflow-hidden",
          isMasked ? "text-muted-foreground bg-white" : "text-white",
          onClick && "cursor-pointer",
        ],
        className,
      )}
    >
      {/* Striped pattern for masked elements */}
      {isMasked && (
        <svg
          className="absolute inset-0 w-full h-full"
          style={{ borderRadius: "inherit" }}
        >
          <defs>
            <pattern
              id={patternId}
              patternUnits="userSpaceOnUse"
              width="6"
              height="6"
            >
              <rect width="6" height="6" fill="transparent" />
              <path
                d="M0,6 L6,0"
                stroke={color}
                strokeWidth="1.5"
                opacity="0.5"
              />
            </pattern>
          </defs>
          <circle cx="50%" cy="50%" r="22" fill={`url(#${patternId})`} />
        </svg>
      )}
      <span className="relative z-10">{children}</span>
    </motion.div>
  );
};

export default Circle;
