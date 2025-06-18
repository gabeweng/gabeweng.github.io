import React from "react";
import '../assets/styles/Main.scss';

const tests = [
  "PSAT: 1520/1520 (National Merit)",
  "SAT: 1600",
  "AP Chemistry 5",
  "AP CS Principles 5",
  "AP US History 5",
  "AP Calculus BC 5",
  "AP Computer Science A 5",
  "AP Microeconomics 5",
  "AP Microeconomics 5",
  "AP Physics C: E&M 5",
  "AP Physics C: Mechanics 5",
  "AP Statistics 5",
  "AP Biology 5",
  "AP Psychology 5",
  "AP Language and Composition 5",
  "AP Government and Politics 5"
];

function Tests() {
  return (
    <div className="items-container" id="tests">
      <h1>Tests</h1>
      <ul className="two-column-list">
        {tests.map((test, idx) => <li key={idx}>{test}</li>)}
      </ul>
    </div>
  );
}

export default Tests; 