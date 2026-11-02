import { useId } from "react";

type LogoProps = {
  size?: number;
  className?: string;
  decorative?: boolean;
};

export function Logo({ size = 32, className, decorative = true }: LogoProps) {
  const rawId = useId();
  const gradientId = `tile-${rawId.replace(/[^a-zA-Z0-9_-]/g, "")}`;

  return (
    <svg
      viewBox="0 0 64 64"
      width={size}
      height={size}
      className={className}
      aria-hidden={decorative ? true : undefined}
      role={decorative ? undefined : "img"}
      aria-label={decorative ? undefined : "RAG Chatbot"}
    >
      <defs>
        <linearGradient
          id={gradientId}
          x1="8"
          y1="2"
          x2="56"
          y2="62"
          gradientUnits="userSpaceOnUse"
        >
          <stop offset="0" stopColor="#3FA2FF" />
          <stop offset="1" stopColor="#0057D9" />
        </linearGradient>
      </defs>
      <rect width="64" height="64" rx="15" fill={`url(#${gradientId})`} />
      <path
        d="M20 11H44A8 8 0 0 1 52 19V35A8 8 0 0 1 44 43H27L14 52L12 43V19A8 8 0 0 1 20 11Z"
        fill="#fff"
        stroke="#fff"
        strokeWidth="2"
        strokeLinejoin="round"
      />
      <g fill="#0B57C4" opacity=".45">
        <rect x="19" y="17" width="8" height="4" rx="2" />
        <rect x="29" y="17" width="12" height="4" rx="2" />
        <rect x="19" y="36" width="7" height="4" rx="2" />
        <rect x="28" y="36" width="9" height="4" rx="2" />
      </g>
      <rect
        x="16.5"
        y="24.5"
        width="31"
        height="9"
        rx="4.5"
        fill="#FFC53D"
      />
      <g fill="#0A3A8A">
        <rect x="19.5" y="27.25" width="9" height="3.5" rx="1.75" />
        <rect x="30.5" y="27.25" width="6" height="3.5" rx="1.75" />
        <rect x="38.5" y="27.25" width="6.5" height="3.5" rx="1.75" />
      </g>
    </svg>
  );
}
