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
    <div className="container" id="courses">
      <h1 className="white-text">Courses</h1>
      <h2 className="white-text">Previous Courses</h2>
      <ul className="two-column-list white-text">
        {previousCourses.map((course, idx) => <li key={idx}>{course}</li>)}
      </ul>
      <h2 className="white-text">Current Courses</h2>
      <ul className="two-column-list white-text">
        {currentCourses.map((course, idx) => <li key={idx}>{course}</li>)}
      </ul>
    </div>
  );
}

export default Courses; 