import React from "react";
import '@fortawesome/free-regular-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faBriefcase, faGraduationCap } from '@fortawesome/free-solid-svg-icons';
import { VerticalTimeline, VerticalTimelineElement }  from 'react-vertical-timeline-component';
import 'react-vertical-timeline-component/style.min.css';
import '../assets/styles/Timeline.scss'

function Timeline() {
  return (
    <div id="history">
      <div className="items-container">
        <h1>Experience & Education</h1>
        <VerticalTimeline>
          <VerticalTimelineElement
            className="vertical-timeline-element--work"
            contentStyle={{ background: 'white', color: 'rgb(39, 40, 34)' }}
            contentArrowStyle={{ borderRight: '7px solid  white' }}
            date="Sep. 2024 - Present"
            iconStyle={{ background: '#5000ca', color: 'rgb(39, 40, 34)' }}
            icon={<FontAwesomeIcon icon={faBriefcase} />}
          >
            <h3 className="vertical-timeline-element-title">Backend Engineer</h3>
            <h4 className="vertical-timeline-element-subtitle">Penn Labs</h4>
            <p>Developed backend ticketing and club management features for Penn Clubs, utilizing a Django-based REST API and integrating with a React/Next.js frontend.</p>
          </VerticalTimelineElement>
          <VerticalTimelineElement
            className="vertical-timeline-element--work"
            date="Sep. 2024 - Present"
            iconStyle={{ background: '#5000ca', color: 'rgb(39, 40, 34)' }}
            icon={<FontAwesomeIcon icon={faBriefcase} />}
          >
            <h3 className="vertical-timeline-element-title">Arete Fellow</h3>
            <h4 className="vertical-timeline-element-subtitle">Penn Effective Altruism</h4>
            <p>Discussed EA concepts such as longtermism, moral circles, cause prioritisation, etc.</p>
          </VerticalTimelineElement>
          <VerticalTimelineElement
            className="vertical-timeline-element--work"
            date="May 2024 - Aug. 2024"
            iconStyle={{ background: '#5000ca', color: 'rgb(39, 40, 34)' }}
            icon={<FontAwesomeIcon icon={faBriefcase} />}
          >
            <h3 className="vertical-timeline-element-title">Software Intern</h3>
            <h4 className="vertical-timeline-element-subtitle">Fermat Capital Management</h4>
            <p>Wrote Data Validation checks, preventing delays in reporting caused by inaccuracies at a hedge fund specializing in cat bonds.</p>
          </VerticalTimelineElement>
          <VerticalTimelineElement
            className="vertical-timeline-element--work"
            date="June 2023 - Aug. 2023"
            iconStyle={{ background: '#5000ca', color: 'rgb(39, 40, 34)' }}
            icon={<FontAwesomeIcon icon={faBriefcase} />}
          >
            <h3 className="vertical-timeline-element-title">Founder</h3>
            <h4 className="vertical-timeline-element-subtitle">Discoveri, LaunchX</h4>
            <p>Created an app connecting students to competitions and teammates. LaunchX is an MIT spinoff with 16% acceptance rate.</p>
          </VerticalTimelineElement>
          <VerticalTimelineElement
            className="vertical-timeline-element--work"
            date="July 2022 - Aug. 2022"
            iconStyle={{ background: '#5000ca', color: 'rgb(39, 40, 34)' }}
            icon={<FontAwesomeIcon icon={faBriefcase} />}
          >
            <h3 className="vertical-timeline-element-title">Research Intern</h3>
            <h4 className="vertical-timeline-element-subtitle">Harvard Coronavirus Visualization Team</h4>
            <p>Invited to join HCVT research team with mainly college students. We analyzed and researched with visualization tools.</p>
          </VerticalTimelineElement>
          <VerticalTimelineElement
            className="vertical-timeline-element--work"
            date="Sep. 2020 - Present"
            iconStyle={{ background: '#5000ca', color: 'rgb(39, 40, 34)' }}
            icon={<FontAwesomeIcon icon={faBriefcase} />}
          >
            <h3 className="vertical-timeline-element-title">Instructor</h3>
            <h4 className="vertical-timeline-element-subtitle">Citizen Invention</h4>
            <p>Led weekly coding enrichment program. Guided 150+ elementary students in coding and robotics, enhancing problem-solving skills. "Employee of 2023"</p>
          </VerticalTimelineElement>
          <VerticalTimelineElement
            className="vertical-timeline-element--education"
            date="Aug. 2024 - Present"
            iconStyle={{ background: '#007bff', color: 'rgb(39, 40, 34)' }}
            icon={<FontAwesomeIcon icon={faGraduationCap} />}
          >
            <h3 className="vertical-timeline-element-title">University of Pennsylvania</h3>
            <h4 className="vertical-timeline-element-subtitle">Class of 2028, Computer Science and Economics</h4>
          </VerticalTimelineElement>
          <VerticalTimelineElement
            className="vertical-timeline-element--education"
            date="Sep. 2020 - June 2024"
            iconStyle={{ background: '#007bff', color: 'rgb(39, 40, 34)' }}
            icon={<FontAwesomeIcon icon={faGraduationCap} />}
          >
            <h3 className="vertical-timeline-element-title">Staples High School</h3>
            <h4 className="vertical-timeline-element-subtitle">Westport, CT</h4>
            <p>Code4ACause co-founder, Fed Challenge Team, Debate Club Secretary, Mu Alpha Theta. GPA: 4.52 (Weighted) / 4.10 (Unweighted)</p>
          </VerticalTimelineElement>
        </VerticalTimeline>
      </div>
    </div>
  );
}

export default Timeline;