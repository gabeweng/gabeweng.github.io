import React from "react";
import GitHubIcon from '@mui/icons-material/GitHub';
import LinkedInIcon from '@mui/icons-material/LinkedIn';
import '../assets/styles/Main.scss';
import gabePhoto from '../assets/images/gabeweng.jpg';

function Main() {
  return (
    <div className="container">
      <div className="about-section">
        <div className="image-wrapper">
          <img src={gabePhoto} alt="Gabriel Weng Avatar" />
        </div>
        <div className="content">
          <div className="social_icons">
            <a href="https://github.com/gabeweng" target="_blank" rel="noreferrer"><GitHubIcon/></a>
            <a href="mailto:gabeweng@gmail.com" target="_blank" rel="noreferrer">Email</a>
            <a href="mailto:gabeweng@sas.upenn.edu" target="_blank" rel="noreferrer">UPenn Email</a>
            <a href="http://gabrielweng.com/" target="_blank" rel="noreferrer"><LinkedInIcon/></a>
          </div>
          <h1>Gabriel Weng</h1>
          <p>University of Pennsylvania | Computer Science & Economics</p>
          <p>An eager learner and problem solver looking for opportunities to apply an analytical mindset to technological challenges. Strong believer in hackathons and open source.</p>
          <div className="mobile_social_icons">
            <a href="https://github.com/gabeweng" target="_blank" rel="noreferrer"><GitHubIcon/></a>
            <a href="mailto:gabeweng@gmail.com" target="_blank" rel="noreferrer">Email</a>
            <a href="mailto:gabeweng@sas.upenn.edu" target="_blank" rel="noreferrer">UPenn Email</a>
            <a href="http://gabrielweng.com/" target="_blank" rel="noreferrer"><LinkedInIcon/></a>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Main;