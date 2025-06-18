import React from "react";
import '@fortawesome/free-regular-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faReact, faPython, faJava } from '@fortawesome/free-brands-svg-icons';
import Chip from '@mui/material/Chip';
import '../assets/styles/Expertise.scss';

const skills = [
    "Debate", "Hackathon", "LLM", "Web", "Chatbot", "OpenAI", "Python", "Java", "NodeJS", "Machine Learning", "Streamlit", "Prompt Engineering", "Cohere", "Velo"
];

const hobbies = [
    "Online Games", "Ukulele", "Pickleball", "Travel & Food"
];

const languages = [
    "English", "Spanish (CT Seal of Biliteracy)"
];

function Expertise() {
    return (
    <div className="container" id="expertise">
        <div className="skills-container">
            <h1>Skills</h1>
            <div className="skills-grid">
                <div className="skill">
                    <FontAwesomeIcon icon={faReact} size="3x"/>
                    <h3>Technical Skills</h3>
                    <div className="flex-chips">
                        {skills.map((label, index) => (
                            <Chip key={index} className='chip' label={label} />
                        ))}
                    </div>
                </div>
                <div className="skill">
                    <FontAwesomeIcon icon={faPython} size="3x"/>
                    <h3>Hobbies</h3>
                    <div className="flex-chips">
                        {hobbies.map((label, index) => (
                            <Chip key={index} className='chip' label={label} />
                        ))}
                    </div>
                </div>
                <div className="skill">
                    <FontAwesomeIcon icon={faJava} size="3x"/>
                    <h3>Languages</h3>
                    <div className="flex-chips">
                        {languages.map((label, index) => (
                            <Chip key={index} className='chip' label={label} />
                        ))}
                    </div>
                </div>
            </div>
        </div>
    </div>
    );
}

export default Expertise;