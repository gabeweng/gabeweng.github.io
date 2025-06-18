import React from "react";
import GitHubIcon from '@mui/icons-material/GitHub';
import LinkedInIcon from '@mui/icons-material/LinkedIn';
import '../assets/styles/Footer.scss'

function Footer() {
  return (
    <footer>
      <div>
        <a href="https://github.com/gabeweng" target="_blank" rel="noreferrer"><GitHubIcon/></a>
        <a href="http://gabrielweng.com/" target="_blank" rel="noreferrer"><LinkedInIcon/></a>
      </div>
      <p>Portfolio of <a href="http://gabrielweng.com/" target="_blank" rel="noreferrer">Gabriel Weng</a></p>
    </footer>
  );
}

export default Footer;