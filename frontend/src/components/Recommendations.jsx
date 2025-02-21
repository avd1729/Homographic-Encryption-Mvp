import React from 'react';

const Recommendations = ({ data }) => {
    return (
        <div>
            <h3>Recommendations:</h3>
            <pre>{data ? JSON.stringify(data, null, 4) : 'No recommendations yet.'}</pre>
        </div>
    );
};

export default Recommendations;
