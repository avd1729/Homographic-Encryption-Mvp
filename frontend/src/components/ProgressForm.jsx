import React from 'react';

const ProgressBar = ({ progress }) => {
    const containerStyle = {
        height: '8px',
        backgroundColor: '#e0e0e0',
        borderRadius: '5px',
        overflow: 'hidden',
        marginTop: '10px',
    };

    const progressBarStyle = {
        width: `${progress}%`,
        height: '100%',
        backgroundColor: '#4CAF50',
        transition: 'width 0.3s ease',
    };

    return (
        <div style={containerStyle}>
            <div style={progressBarStyle}></div>
        </div>
    );
};

export default ProgressBar;
