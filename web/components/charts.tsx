"use client";

import { memo } from "react";
import { cn } from "../lib/utils";

type HorizontalBarDatum = {
  id: string;
  label: string;
  value: number;
  emphasis?: boolean;
};

interface HorizontalBarChartProps {
  title: string;
  caption?: string;
  data: HorizontalBarDatum[];
  maxValue?: number;
  direction?: "ltr" | "rtl";
}

export const HorizontalBarChart = memo(function HorizontalBarChart({
  title,
  caption,
  data,
  maxValue,
  direction = "ltr",
}: HorizontalBarChartProps) {
  const max = maxValue ?? Math.max(...data.map((item) => item.value));
  return (
    <figure className={cn("chart-card", direction === "rtl" && "rtl")}>
      <figcaption>
        <strong>{title}</strong>
        {caption ? <p className="chart-caption">{caption}</p> : null}
      </figcaption>
      <div className="chart-bars">
        {data.map((item) => (
          <div key={item.id} className="chart-bar-row">
            <span className="chart-label">
              <span className={cn("chart-dot", item.emphasis && "chart-dot--emphasis")} aria-hidden="true" />
              {item.label}
            </span>
            <div className="chart-bar-track" aria-hidden="true">
              <div
                className={cn("chart-bar-fill", item.emphasis && "chart-bar-fill--emphasis")}
                style={{
                  width: `${Math.max(6, (item.value / max) * 100)}%`,
                }}
              />
            </div>
            <span className="chart-value">{item.value.toFixed(1)}</span>
          </div>
        ))}
      </div>
    </figure>
  );
});

type LinePoint = { x: number; y: number };

interface DualLineChartProps {
  title: string;
  caption?: string;
  retainedSeries: LinePoint[];
  churnSeries: LinePoint[];
  retainedLabel: string;
  churnLabel: string;
  direction?: "ltr" | "rtl";
}

const buildSmoothPath = (points: LinePoint[], maxX: number, maxY: number) => {
  if (points.length === 0) return "";
  const scaled = points.map((p) => ({
    x: (p.x / maxX) * 100,
    y: 100 - (p.y / maxY) * 100,
  }));

  return scaled
    .map((point, idx) => {
      if (idx === 0) {
        return `M ${point.x},${point.y}`;
      }
      const prev = scaled[idx - 1];
      const controlX = (prev.x + point.x) / 2;
      return `C ${controlX},${prev.y} ${controlX},${point.y} ${point.x},${point.y}`;
    })
    .join(" ");
};

const buildAreaPath = (linePath: string) => {
  if (!linePath) return "";
  return `${linePath} L 100,100 L 0,100 Z`;
};

export const DualLineChart = memo(function DualLineChart({
  title,
  caption,
  retainedSeries,
  churnSeries,
  retainedLabel,
  churnLabel,
  direction = "ltr",
}: DualLineChartProps) {
  const allPoints = [...retainedSeries, ...churnSeries];
  const maxX = Math.max(...allPoints.map((p) => p.x));
  const maxY = Math.max(...allPoints.map((p) => p.y));
  const viewBoxPadding = 8;

  const retainedPath = buildSmoothPath(retainedSeries, maxX, maxY);
  const churnPath = buildSmoothPath(churnSeries, maxX, maxY);

  return (
    <figure className={cn("chart-card", direction === "rtl" && "rtl")}>
      <figcaption>
        <strong>{title}</strong>
        {caption ? <p className="chart-caption">{caption}</p> : null}
      </figcaption>
      <div className="chart-line-wrapper">
        <svg viewBox={`0 0 ${100 + viewBoxPadding} ${100 + viewBoxPadding}`} preserveAspectRatio="none">
          <defs>
            <linearGradient id="retained-gradient" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stopColor="rgba(27,27,27,0.25)" />
              <stop offset="100%" stopColor="rgba(27,27,27,0.03)" />
            </linearGradient>
            <linearGradient id="churn-gradient" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stopColor="rgba(90,90,90,0.32)" />
              <stop offset="100%" stopColor="rgba(90,90,90,0.06)" />
            </linearGradient>
          </defs>
          <g transform={`translate(${viewBoxPadding / 2},${viewBoxPadding / 2})`}>
            {[25, 50, 75].map((y) => (
              <polyline
                key={y}
                points={`0,${y} 100,${y}`}
                stroke="rgba(27,27,27,0.1)"
                strokeWidth="0.8"
                fill="none"
              />
            ))}
            <polyline points="0,100 100,100" stroke="rgba(27,27,27,0.18)" strokeWidth="1" fill="none" />

            {churnPath && (
              <>
                <path
                  d={buildAreaPath(churnPath)}
                  fill="url(#churn-gradient)"
                  stroke="none"
                  opacity={0.65}
                />
                <path d={churnPath} stroke="rgba(90,90,90,0.95)" strokeWidth="2.2" fill="none" strokeLinejoin="round" strokeLinecap="round" />
              </>
            )}
            {retainedPath && (
              <>
                <path
                  d={buildAreaPath(retainedPath)}
                  fill="url(#retained-gradient)"
                  stroke="none"
                  opacity={0.55}
                />
                <path d={retainedPath} stroke="rgba(27,27,27,0.9)" strokeWidth="2.2" fill="none" strokeLinejoin="round" strokeLinecap="round" />
              </>
            )}
          </g>
        </svg>
        <div className="chart-legend">
          <span className="legend-item">
            <span className="legend-swatch legend-swatch--retained" />
            <span>{retainedLabel}</span>
          </span>
          <span className="legend-item">
            <span className="legend-swatch legend-swatch--churn" />
            <span>{churnLabel}</span>
          </span>
        </div>
      </div>
    </figure>
  );
});
