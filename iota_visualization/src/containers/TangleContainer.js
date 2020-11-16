import React from 'react';
import PropTypes from 'prop-types';
import Tangle from '../components/Tangle';
import {connect} from 'react-redux';
import * as d3Force from 'd3-force';
import {scaleLinear} from 'd3-scale';
import {loadPeer, loadTangle, numberOfPeers} from '../shared/generateData';
import Slider from 'rc-slider';
import Tooltip from 'rc-tooltip';
import 'rc-slider/assets/index.css';
import 'rc-tooltip/assets/bootstrap.css';
import {getAncestors, getDescendants, getTips} from '../shared/algorithms';
import './radio-button.css';
import '../components/Tangle.css';
import {Form, Header} from "semantic-ui-react";

const mapStateToProps = (state, ownProps) => ({});
const mapDispatchToProps = (dispatch, ownProps) => ({});

const nodeRadiusMax = 25;
const nodeRadiusMin = 13;
const showLabelsMinimumRadius = 15;
const getNodeRadius = nodeCount => {
    const smallNodeCount = 20;
    const largeNodeCount = 100;

    if (nodeCount < smallNodeCount) {
        return nodeRadiusMax;
    }
    if (nodeCount > largeNodeCount) {
        return nodeRadiusMin;
    }
    const scale = scaleLinear().domain([smallNodeCount, largeNodeCount]);
    scale.range([nodeRadiusMax, nodeRadiusMin]);

    return scale(nodeCount);
};

const leftMargin = 10;
const rightMargin = 10;
const bottomMargin = 111;

const iterationDefault = 1;// (window.location.hash && parseInt(window.location.hash.slice(1))) || 67;

const Handle = Slider.Handle;
const sliderHandle = props => {
    const {value, dragging, index, ...restProps} = props;
    return (
        <Tooltip
            prefixCls='rc-slider-tooltip'
            overlay={value}
            visible={dragging}
            placement='top'
            key={index}
        >
            <Handle value={value} {...restProps} />
        </Tooltip>
    );
};

sliderHandle.propTypes = {
    value: PropTypes.number.isRequired,
    dragging: PropTypes.bool.isRequired,
    index: PropTypes.number.isRequired,
};

const TipAlgorithmLabel = ({selectedAlgorithm, onChange, algoKey}) =>
    <label className='container' key={algoKey}>
        <div style={{fontSize: 10}}>
            {tipSelectionDictionary[algoKey].label}
        </div>
        <input type='radio' name='radio' value={algoKey}
               checked={selectedAlgorithm === algoKey}
               onChange={onChange}
        />
        <span className='checkmark'></span>
    </label>;

TipAlgorithmLabel.propTypes = {
    selectedAlgorithm: PropTypes.string.isRequired,
    onChange: PropTypes.any,
    algoKey: PropTypes.string.isRequired,
};


class TangleContainer extends React.Component {
    constructor(props) {
        super();

        this.state = {
            nodes: [],
            links: [],
            iteration: iterationDefault,
            width: 300, // default values
            height: window.innerHeight,
            nodeRadius: getNodeRadius(20 /* = nodeCountDefault */),
            peerId: null
        };
        this.updateWindowDimensions = this.updateWindowDimensions.bind(this);

        this.force = d3Force.forceSimulation();
        this.force.alphaDecay(0.1);

        this.force.on('tick', () => {
            this.force.nodes(this.state.nodes);

            // restrict nodes to window area
            for (let node of this.state.nodes) {
                node.y = Math.max(this.state.nodeRadius, Math.min(this.state.height - this.state.nodeRadius, node.y));
            }

            this.setState({
                links: this.state.links,
                nodes: this.state.nodes,
            });
        });

        this.max_peers = 2;
        numberOfPeers().then(p => {
            console.log("test");
            console.log(p);
            this.max_peers = p;
        }
            );
    }

    componentWillUnmount() {
        this.force.stop();
        window.removeEventListener('resize', this.updateWindowDimensions);
    }

    componentDidMount() {
        this.startNewTangle();
        let number = this.getSelectedPeerNumber()
        this.loadPeer(number)
        this.updateWindowDimensions();
        window.addEventListener('resize', this.updateWindowDimensions);
    }

    updateWindowDimensions() {
        this.setState({
            width: window.innerWidth - leftMargin - rightMargin,
            height: window.innerHeight - bottomMargin,
        }, () => {
            this.recalculateFixedPositions();
            this.force
                .force('no_collision', d3Force.forceCollide().radius(this.state.nodeRadius * 2).strength(0.01).iterations(15))
                .force('pin_y_to_center', d3Force.forceY().y(d => this.state.height / 2).strength(0.1))
                .force('pin_x_to_time', d3Force.forceX().x(d => this.xFromTime(d.time)).strength(1))
                .force('link', d3Force.forceLink().links(this.state.links).strength(0.5).distance(this.state.nodeRadius * 3)); // strength in [0,1]

            this.force.restart().alpha(1);
        });
    }

    async startNewTangle() {
        const tangle = await loadTangle({peerNumber: this.state.iteration});
        const nodeRadius = getNodeRadius(tangle.nodes.length);

        const {width, height} = this.state;

        for (let node of tangle.nodes) {
            node.y = height / 4 + Math.random() * (height / 2),
                node.x = width / 2; // required to avoid annoying errors
        }

        this.force.stop();

        this.setState({
            nodes: tangle.nodes,
            links: tangle.links,
            nodeRadius,
        }, () => {
            // Set all nodes' x by time value after state has been set
            this.recalculateFixedPositions();
        });

        this.force.restart().alpha(1);
    }

    recalculateFixedPositions() {
        if (this.state.nodes.length === 0)
            return;

        // Set genesis's y to center
        const genesisNode = this.state.nodes[0];
        genesisNode.fx = this.state.height / 2;

        for (let node of this.state.nodes) {
            node.fx = this.xFromTime(node.time);
        }
    }

    xFromTime(time) {
        const padding = this.state.nodeRadius;
        // Avoid edge cases with 0 or 1 nodes
        if (this.state.nodes.length < 2) {
            return padding;
        }

        const maxTime = this.state.nodes[this.state.nodes.length - 1].time;

        // Rescale nodes' x to cover [margin, width-margin]
        const scale = scaleLinear().domain([0, maxTime]);
        scale.range([padding, this.state.width - padding]);

        return scale(time);
    }

    mouseEntersNodeHandler(e) {
        const name = e.target.getAttribute('name');
        this.setState({
            hoveredNode: this.state.nodes.find(node => node.name === name),
        });
    }

    mouseLeavesNodeHandler(e) {
        this.setState({
            hoveredNode: undefined,
        });
    }

    getApprovedNodes(root) {
        if (!root) {
            return {nodes: new Set(), links: new Set()};
        }

        return getDescendants({
            nodes: this.state.nodes,
            links: this.state.links,
            root,
        });
    }

    getApprovingNodes(root) {
        if (!root) {
            return {nodes: new Set(), links: new Set()};
        }

        return getAncestors({
            nodes: this.state.nodes,
            links: this.state.links,
            root,
        });
    }

    getSelectedPeerNumber() {
        return document.getElementById("peerNumberInput").value
    }

    async loadPeer(peerNumber) {
        const peer = await loadPeer({peerNumber: peerNumber});
        this.state.peerId = peer.client_id;
    }

    render() {
        const {width, height} = this.state;
        const approved = this.getApprovedNodes(this.state.hoveredNode);
        const approving = this.getApprovingNodes(this.state.hoveredNode);
        
    

        return (
            <div>
                <div className='top-bar-container' style={{width}}>
                    <div className='left-cell'></div>
                    <div className='right-cell'></div>
                    <div className='top-bar-row'>
                        <div className='slider-title'>
                            <Header as='h4' textAlign='center'>
                                Peer ({this.state.peerId})
                            </Header>
                        </div>
                        <div className='slider-container'>
                            <Form size='mini' key='mini'
                                  onSubmit={() => {
                                      let number = this.getSelectedPeerNumber()
                                      this.loadPeer(number)
                                      this.setState(Object.assign(this.state, {number}))
                                      this.startNewTangle();
                                  }}>
                                <Form.Group>
                                    <Form.Input fluid
                                                id="peerNumberInput"
                                                type="number"
                                                defaultValue={1}
                                                min={1}
                                                max={this.max_peers}
                                                action={{icon: 'refresh'}}
                                                width={2}
                                                onChange={() => {
                                                    let number = this.getSelectedPeerNumber()
                                                    this.loadPeer(number)
                                                    this.setState(Object.assign(this.state, {number}));
                                                    this.startNewTangle();
                                                }}/>
                                </Form.Group>
                            </Form>
                        </div>
                    </div>
                </div>
                <Tangle links={this.state.links} nodes={this.state.nodes}
                        nodeCount={6}
                        width={width}
                        height={height}
                        leftMargin={leftMargin}
                        rightMargin={rightMargin}
                        nodeRadius={this.state.nodeRadius}
                        mouseEntersNodeHandler={this.mouseEntersNodeHandler.bind(this)}
                        mouseLeavesNodeHandler={this.mouseLeavesNodeHandler.bind(this)}
                        approvedNodes={approved.nodes}
                        approvedLinks={approved.links}
                        approvingNodes={approving.nodes}
                        approvingLinks={approving.links}
                        hoveredNode={this.state.hoveredNode}
                        selectedPeerId={this.state.peerId}
                        tips={getTips({
                            nodes: this.state.nodes,
                            links: this.state.links,
                        })}
                        showLabels={this.state.nodeRadius > showLabelsMinimumRadius ? true : false}
                />
            </div>
        );
    }
}

const TangleContainerConnected = connect(
    mapStateToProps,
    mapDispatchToProps
)(TangleContainer);

export default TangleContainerConnected;
