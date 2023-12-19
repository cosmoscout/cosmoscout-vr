////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

// The TimeInterval has a single output socket and a custom widget for entering a date and time. The
// custom widget is defined further below.
// The TimeIntervalComponent serves as a kind of factory. Whenever a new node is created, the
// builder() method is called. It is required that the class is called <NAME>Component.
class TimeIntervalComponent extends Rete.Component {
  constructor() {
    // This name must match the TimeInterval::sName defined in TimeInterval.cpp.
    super("TimeInterval");

    // This specifies the submenu from which this node can be created in the node editor.
    this.category = "Constants";
  }

  getDisplayDate(dateText) {
    const date = new Date(dateText);
    let result;

    result = date.getUTCDate() + "." + (date.getUTCMonth() + 1) + "." + date.getUTCFullYear() + 
      " " + date.getUTCHours() + ":" + (date.getUTCMinutes().length == 1 ? "0" + date.getUTCMinutes() : date.getUTCMinutes())
      + " (UTC)";
    return result;
  }

  // Called whenever a new node of this type needs to be constructed.
  builder(node) {

    // This node has a single output. The first parameter is the name of the socket and must be
    // unique amongst all sockets. It is also used in the TimeIntervalComponent::process() to write
    // the output of this node. The second parameter is shown as name on the node. The last
    // parameter references a socket type which has been registered with the node factory
    // before.
    let input = new Rete.Input('timeIntervalsIn', "Time Intervals", CosmoScout.socketTypes['WCSTimeIntervals']);
    node.addInput(input);

    let output = new Rete.Output('value', "Time", CosmoScout.socketTypes['WCSTime']);
    node.addOutput(output);
    
    // Add the TimeInterval input widget. The name parameter must be unique
    // amongst all controls of this node. The NumberControl class is defined further below.
    let timeIntervalControl = new TimeIntervalControl('TimeInterval');

    let intervalDropDown = new DropDownControl('selectInterval', (newInterval) => {
      timeIntervalControl.setRangeStart(new Date(node.data.intervals[newInterval.value].split("/")[0]).toISOString());
      timeIntervalControl.setRangeEnd(new Date(node.data.intervals[newInterval.value].split("/")[1]).toISOString());
      // send new range to cpp TODO
      CosmoScout.sendMessageToCPP({interval: timeIntervalControl.getRange()}, node.id);
    }, "Interval", [{value: 0, text: "None"}]);
    
    node.addControl(intervalDropDown);
    node.addControl(timeIntervalControl);

    // Once the HTML element for this node has been created, the node.onInit() method will be
    // called. This is used here to initialize the input widget. The node.data object may
    // contain a TimeInterval as returned by TimeInterval::getData() which - if present - should be
    // preselected.
    node.onInit = (nodeDiv) => { 
      timeIntervalControl.init(nodeDiv, node.data); 
      
      intervalDropDown.init(nodeDiv, {
          options: node.data.intervals?.map((interval, index) => ({
          value: index,
          text: interval
        })),
        selectedValue: node.data.selectedInterval
      });
    };

    node.onMessageFromCPP = (message) => {
      if (message["intervals"]) {
        let intervals = [];

        for (let i = 0; i < message["intervals"].length ; i++) {
          const dates = message["intervals"][i].split("/");
          const dateText = this.getDisplayDate(dates[0]) + " - " + this.getDisplayDate(dates[1]);
          intervals.push({value: i, text: dateText});
        } 

        node.data.intervals = message["intervals"];
        intervalDropDown.setOptions(intervals);
      }
    }

    return node;
  }
}

// This is the widget which is used for inserting the number.
class TimeIntervalControl extends Rete.Control {
  constructor(key) {
    super(key);
    
    this.id = crypto.randomUUID();

    // This HTML code will be used whenever a node is created with this widget.
    this.template = `
    <div class="container-fluid" style="width: 250px">
          <div class="row">
            <div class="col-2">Start:</div>
            <input class="offset-1 col-3 TimeInterval-input" name="dateTimeStart-${this.id}" id="dateTimeStart-${this.id}" type="dateTime-local" />
          </div>
          <br>
          <div class="row">
            <div class="col-2">End:</div>
            <input class="offset-1 col-3 TimeInterval-input" name="dateTimeEnd-${this.id}" id="dateTimeEnd-${this.id}" type="dateTime-local" />
          </div>
        </div>

          <style>
            .TimeInterval-input {
              max-width: 100%; !important
              width: 170px !important;
            }
          </style>
        `;
  }

  // This is called by the node.onInit() above once the HTML element for the node has been
  // created. If present, the data object may contain a number as returned by
  // TimeInterval::getData() which - if present - should be preselected.
  init(nodeDiv, data) {
    // Get our input element.
    let inputs = nodeDiv.querySelectorAll("input");
    this.elStart = inputs[0];
    this.elEnd = inputs[1];

    // Preselect a number if one was given.
    if (data.value) {
      // this.el.value = data.value;
    }

    // Send an update to the node editor server whenever the user enters a new value.
    this.elStart.addEventListener(
      'input', e => { 
        CosmoScout.sendMessageToCPP(this.getRange(), this.parent.id);
    }); 
    this.elEnd.addEventListener(
      'input', e => { 
        CosmoScout.sendMessageToCPP(this.getRange(), this.parent.id)
    }); 

    // Stop propagation of pointer move events. Else we would drag the node if we tried to
    // select some text in the input field.
    this.elStart.addEventListener('pointermove', e => e.stopPropagation());
    this.elEnd.addEventListener('pointermove', e => e.stopPropagation());
  }

  setRangeStart(start) {
    this.elStart.min = start;
    this.elEnd.min = start;
  }

  setRangeEnd(end) {
    this.elStart.max = end;
    this.elEnd.max = end;
  }

  getRange() {
    if (this.elEnd.value) {
      return this.elStart.value + "/" + this.elEnd.value;
    }
    return this.elStart.value;
  }
}
