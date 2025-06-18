import React from "react";
import '../assets/styles/Main.scss';

const previousCourses = [
  "Mathematical Foundations of Computer Science",
  "Programming Languages And Techniques I",
  "Privacy and Surveillance",
  "Probability"
];

const currentCourses = [
  "Programming Languages And Techniques II",
  "Introduction to Computer Systems",
  "Linear Algebra / Differential Equations",
  "Statistical Inference",
  "International Economics"
];

function Courses() {
  return (
    <div className="items-container" id="courses">
      <h1>Courses</h1>
      <h2>Previous Courses</h2>
      <ul className="two-column-list">
        {previousCourses.map((course, idx) => <li key={idx}>{course}</li>)}
      </ul>
      <h2>Current Courses</h2>
      <ul className="two-column-list">
        {currentCourses.map((course, idx) => <li key={idx}>{course}</li>)}
      </ul>
    </div>
  );
}

export default Courses; 