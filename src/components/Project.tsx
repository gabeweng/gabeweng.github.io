import React from "react";
import '../assets/styles/Project.scss';
import mock01 from '../assets/images/mock01.png';
import mock02 from '../assets/images/mock02.png';
import mock03 from '../assets/images/mock03.png';
import mock04 from '../assets/images/mock04.png';

function Project() {
    return(
    <div className="projects-container" id="projects">
        <h1>Projects, Awards & Publications</h1>
        <div className="projects-grid">
            <div className="project">
                <img src={mock01} alt="Discoveri (LaunchX)" className="project-img" />
                <h2>Discoveri (LaunchX)</h2>
                <p>Founder. App connects students to competitions and teammates based on interests, locations, and subjects. LaunchX is an MIT spinoff with 16% acceptance rate.</p>
            </div>
            <div className="project">
                <img src={mock02} alt="Penn Labs - Penn Clubs" className="project-img" />
                <h2>Penn Labs - Penn Clubs</h2>
                <p>Backend Engineer. Developed backend ticketing and club management features for Penn Clubs, utilizing a Django-based REST API and integrating with a React/Next.js frontend.</p>
            </div>
            <div className="project">
                <img src={mock03} alt="Harvard Coronavirus Visualization Team" className="project-img" />
                <h2>Harvard Coronavirus Visualization Team</h2>
                <p>Research Intern. Analyzed and researched with visualization tools as part of a team of mainly college students.</p>
            </div>
            <div className="project">
                <img src={mock04} alt="Citizen Invention" className="project-img" />
                <h2>Citizen Invention</h2>
                <p>Instructor. Led weekly coding enrichment program. Guided 150+ elementary students in coding and robotics, enhancing problem-solving skills. "Employee of 2023"</p>
            </div>
        </div>
        <h2>Awards</h2>
        <ul>
            <li><a href="https://www.lockheedmartin.com/en-us/who-we-are/communities/codequest/code-quest-past-quests/codequest-2022.html" target="_blank" rel="noreferrer">1st Place, Advanced Div, Lockheed Martin Code Quest "Sandy Coder"</a></li>
            <li><a href="https://drive.google.com/file/d/1NVCmX6Czi4QgtQO7aOIuDq-b9ds_6XAq/view?usp=share_link" target="_blank" rel="noreferrer">Semi-finalist, Modeling the Future, Actuarial Foundation 2023</a></li>
            <li>Finalist, Harvard Pre-Collegiate Economics Challenge 2023</li>
            <li><a href="https://devpost.com/software/aesculapius" target="_blank" rel="noreferrer">1st Place, $1000 Aesculapius Healthcare Track, MetroHack 2022</a></li>
            <li><a href="https://devpost.com/software/ciceroai" target="_blank" rel="noreferrer">1st Place, CiceroAI (NLP, Education), ROBOHackIT 2022</a></li>
            <li><a href="https://rosalind.info/users/darrenweng/" target="_blank" rel="noreferrer">US top 68, Rosalind Bioinformatics</a></li>
            <li><a href="https://devpost.com/software/pridesum" target="_blank" rel="noreferrer">Most Innovative Award, Pridesum (Web3), PrideHack II, 2022</a></li>
            <li><a href="https://www.newyorkfed.org/medialibrary/media/outreach-and-education/hsfc-book-2022-final-online-version-small" target="_blank" rel="noreferrer">Winner, New York Fed Challenge, Top teams in 'Future Economists' 2022</a></li>
            <li><a href="https://www.challenge.gov/?challenge=brain-initiative-2022&tab=winners" target="_blank" rel="noreferrer">Winner, NIH BRAIN Initiative Challenge 2022</a></li>
            <li><a href="https://devpost.com/software/chiron-bxvirh" target="_blank" rel="noreferrer">3rd Place, Chiron (Quizlet, Chatbot), RoboHacks 2, 2022</a></li>
            <li><a href="https://devpost.com/software/collegeviz" target="_blank" rel="noreferrer">Wolfram Award, CollegeViz Vizathon, Harvard 2021</a></li>
            <li><a href="https://robotbenchmark.net/gabrielweng" target="_blank" rel="noreferrer">Ranked top 8%, Robot Benchmark, 2023</a></li>
            <li>RIT Computing Award, Staples High School $76,000 scholarship. 2023</li>
        </ul>
        <h2>Publications</h2>
        <ul id="publications">
            <li>The Effect of Covid-19 Pandemic on Food Insecurity, Harvard CVT, Aug. 2022</li>
            <li>Ethical Considerations of Brain Technologies, NIH Brain Institute, April 2022</li>
            <li>Metaverse Virtualization as a Solution for Climate Change, Federal Reserve Bank of New York, Feb. 2022</li>
            <li>Overdose Crisis — Trends, Policies, and Mitigation, The Actuarial Foundation, Feb. 2023</li>
            <li>Treating T1 Diabetes with Mesenchymal Stem Cells, Yale Young Global Scholar IST, June 2023</li>
        </ul>
    </div>
    );
}

export default Project;