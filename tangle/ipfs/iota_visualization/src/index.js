import React from 'react';
import {render} from 'react-dom';
import reducer from './reducer';
import {Provider} from 'react-redux';
import {createStore} from 'redux';
import TangleContainer from './containers/TangleContainer';

window.addEventListener('load', function () {
    const store = createStore(reducer, window.__REDUX_DEVTOOLS_EXTENSION__ && window.__REDUX_DEVTOOLS_EXTENSION__());

    render(
        <Provider store={store}>
            <div>
                <div className='title-bar-container'>
                    <div className='left-title'>
                        <h2>Tangle Visualization</h2>
                    </div>
                    <div className='right-title'>
                    </div>
                </div>
                <TangleContainer/>
            </div>
        </Provider>,
        document.getElementById('container')
    );
});
