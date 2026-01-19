"use client";

import InputOutput from "../shared/input-output";
import Jacobian from "../shared/jacobian";

const ElementwiseIndependent = () => {
  return (
    <div className="flex flex-col">
      <div className="flex flex-row justify-between gap-2 my-8">
        <InputOutput />
        <Jacobian />
      </div>
    </div>
  );
};

export default ElementwiseIndependent;
