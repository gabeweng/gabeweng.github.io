import React from "react";
import '../assets/styles/Main.scss';

const certificates = [
  { name: "Git & Github (MIT Lincoln Lab)", url: "https://courses.bwsix.edly.io/certificates/694433ee551c4c85a5b1b933b5477b0c" },
  { name: "Cybersecurity (Lincoln Lab)", url: "https://courses.bwsix.edly.io/certificates/38f914d7661449ac86ff3df5ae81e81c" },
  { name: "Python Core (MIT Lincoln Lab)", url: "https://courses.bwsix.edly.io/certificates/661b47e727f24a52be3afec2e72ee968" },
  { name: "AI Scholars (Inspirit AI)", url: "https://drive.google.com/file/d/1rkbdAuCOLiyYDO_EEUX8giSOZyHR4p2n/view?usp=share_link" },
  { name: "Fundamentals of Quantitative Modeling (Wharton)", url: "https://coursera.org/share/10713b798e3137bc79ac1560d518e15a" },
  { name: "Narrative Economics (Yale)", url: "https://coursera.org/share/f29a71ad93be1ed992219e2cc470186e" },
  { name: "Introduction to Generative AI (Google)", url: "https://www.cloudskillsboost.google/public_profiles/6eb9846a-b12e-48dd-9000-93019d0155bb/badges/3992433" },
  { name: "Introduction to Large Language Models (Google)", url: "https://www.cloudskillsboost.google/public_profiles/6eb9846a-b12e-48dd-9000-93019d0155bb/badges/3993692" },
];

function Certificates() {
  return (
    <div className="container" id="certificates">
      <h1 className="white-text">Certificates</h1>
      <ul className="two-column-list white-text">
        {certificates.map((cert, idx) => (
          <li key={idx}>
            <a href={cert.url} target="_blank" rel="noreferrer">{cert.name}</a>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default Certificates; 