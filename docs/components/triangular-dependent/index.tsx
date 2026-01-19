"use client";

import { useSetAtom } from "jotai";
import { useEffect } from "react";
import InputOutput from "../shared/input-output";
import Jacobian from "../shared/jacobian";
import { patternAtom } from "../shared/atoms";

const TriangularDependent = () => {
  const setPattern = useSetAtom(patternAtom);

  useEffect(() => {
    setPattern("lower-triangular");
    return () => setPattern("diagonal"); // Reset on unmount
  }, [setPattern]);

  return (
    <div className="flex flex-col">
      <div className="flex flex-row justify-between gap-2 my-8">
        <InputOutput />
        <Jacobian />
      </div>
    </div>
  );
};

export default TriangularDependent;
