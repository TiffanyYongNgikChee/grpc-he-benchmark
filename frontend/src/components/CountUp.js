import { useEffect, useRef, useState } from "react";

/**
 * CountUp — Animates a number from 0 to `end` over `duration` ms.
 *
 * Props:
 *  - end: number        (target value)
 *  - duration: number   (animation length in ms, default 800)
 *  - decimals: number   (decimal places, default 1)
 *  - suffix: string     (appended after the number, e.g. "%" or "ms")
 */
export default function CountUp({ end, duration = 800, decimals = 1, suffix = "" }) {
  const [display, setDisplay] = useState(0);
  const rafRef = useRef(null);
  const startRef = useRef(null);

  useEffect(() => {
    startRef.current = null;

    function step(timestamp) {
      if (!startRef.current) startRef.current = timestamp;
      const elapsed = timestamp - startRef.current;
      const progress = Math.min(elapsed / duration, 1);

      // ease-out cubic for a satisfying deceleration
      const eased = 1 - Math.pow(1 - progress, 3);
      setDisplay(eased * end);

      if (progress < 1) {
        rafRef.current = requestAnimationFrame(step);
      }
    }

    rafRef.current = requestAnimationFrame(step);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [end, duration]);

  return (
    <>
      {display.toFixed(decimals)}{suffix}
    </>
  );
}
