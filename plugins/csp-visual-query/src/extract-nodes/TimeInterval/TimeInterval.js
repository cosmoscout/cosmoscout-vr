////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

/**
 * Converts a dateTime string in the ISO 8601 Format to a string that matches the style of
 * the CosmoScout time format.
 * @param {string} dateText dateTime in ISO 8601 to convert
 * @returns dateTime in CosmoScout format "YYYY-MM-DD HH-MM-SS"
 */
function convertToReadableDate(dateText) {
  let dateTime = dateText.split(".")[0];
  dateTime     = dateTime.split("T");
  return dateTime[0] + " " + dateTime[1];
}

// The TimeInterval has a single output socket and a custom widget for entering a date and time. The
// custom widget is defined further below.
// The TimeIntervalComponent serves as a kind of factory. Whenever a new node is created, the
// builder() method is called. It is required that the class is called <NAME>Component.
class TimeIntervalComponent extends Rete.Component {
  constructor() {
    // This name must match the TimeInterval::sName defined in TimeInterval.cpp.
    super("TimeInterval");

    // This specifies the submenu from which this node can be created in the node editor.
    this.category = "Data Extraction";
  }

  // Called whenever a new node of this type needs to be constructed.
  builder(node) {

    // This node has a single output. The first parameter is the name of the socket and must be
    // unique amongst all sockets. It is also used in the TimeIntervalComponent::process() to write
    // the output of this node. The second parameter is shown as name on the node. The last
    // parameter references a socket type which has been registered with the node factory
    // before.
    let input = new Rete.Input(
        'timeIntervalsIn', "Time Intervals", CosmoScout.socketTypes['WCSTimeIntervals']);
    node.addInput(input);

    let output = new Rete.Output('value', "Time", CosmoScout.socketTypes['WCSTime']);
    node.addOutput(output);

    // Add the TimeInterval input widget. The name parameter must be unique
    // amongst all controls of this node. The TimeIntervalControl class is defined further below.
    let timeIntervalControl = new TimeIntervalControl('TimeInterval');

    let intervalDropDown = new DropDownControl('selectInterval', (newInterval) => {
      // new interval select:
      const selectedIndex =
          intervalDropDown.getSelectedIndex() - 1; // subtracting one because the displayed dropdown
                                                   // list includes an additional "none" element
      if (selectedIndex != -1) {
        timeIntervalControl.setStepSize(node.data.intervals[selectedIndex]["step"]["size"],
            node.data.intervals[selectedIndex]["step"]["unit"]);

      } else { // "None" selected
        timeIntervalControl.setStepSize("", "");
        timeIntervalControl.setCurrentTime("");
      }
      // send new time interval to cpp
      CosmoScout.sendMessageToCPP({intervalIndex: selectedIndex}, node.id);
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
        options: node.data.intervals?.map((interval, index) => ({value: index, text: interval})),
        selectedValue: node.data.selectedIntervalIndex
      });
    };

    node.onMessageFromCPP =
        (message) => {
          // intervals input changed
          if (message["intervals"]) {
            let intervals = [{value: 0, text: "None"}];

            timeIntervalControl.setStepSize("", "");
            timeIntervalControl.setCurrentTime("");

            if (message["intervals"] != "reset") {
              for (let i = 0; i < message["intervals"].length; i++) {
                const dateText = convertToReadableDate(message["intervals"][i]["start"]) + " - " +
                                 convertToReadableDate(message["intervals"][i]["end"]);
                intervals.push({value: i + 1, text: dateText});
              }
            }

            intervalDropDown.setOptions(intervals);
            node.data.intervals = message["intervals"];
          }

          // interval or time step changed
          if (message["currentTime"]) {
            if (message["currentTime"] == "reset") {
              timeIntervalControl.setCurrentTime("");
            } else {
              timeIntervalControl.setCurrentTime(convertToReadableDate(message["currentTime"]));
            }
          }
        }

    return node;
  }
}

class TimeIntervalControl extends Rete.Control {
  constructor(key) {
    super(key);

    this.id = crypto.randomUUID();

    // This HTML code will be used whenever a node is created with this widget.
    this.template = `
      <div class="container-fluid">
        <div class="row mb-1">
          <div class="col">step size:</div>
          <div class="col" id="${this.id}-stepSize"></div>
        </div>
        <hr>
        <div class="row mb-1">
          <input class="col" style="display: flex" type="checkbox" id="${
        this.id}-syncSimulationTime" name="${this.id}-syncSimulationTime" />
          <label for="${this.id}-syncSimulationTime">sync with simulation time</label>
        </div>

        <div class="row mb-1">
          <div class="col-3">
            <button class="btn glass block" data-toggle="tooltip" title="First" id="${
        this.id}-firstStep">
              <i class="material-icons">skip_previous</i>
            </button>
          </div>
          <div class="col-3">
            <button class="btn glass block" data-toggle="tooltip" title="Previous" id="${
        this.id}-prevStep">
              <i class="material-icons">navigate_before</i>
            </button>
          </div>
          <div class="col-3">
            <button class="btn glass block" data-toggle="tooltip" title="Next" id="${
        this.id}-nextStep">
              <i class="material-icons">navigate_next</i>
            </button>
          </div>
          <div class="col-3">
            <button class="btn glass block" data-toggle="tooltip" title="Last" id="${
        this.id}-lastStep">
              <i class="material-icons">skip_next</i>
            </button>
          </div>
        </div>

        <div class="row mb-1">
          <div class="col-md-12 mx-auto" id="${this.id}-currentTime"></div>
        </div>
        </div>
        `;
  }

  // This is called by the node.onInit() above once the HTML element for the node has been
  // created. If present, the data object may contain a number as returned by
  // TimeInterval::getData() which - if present - should be preselected.
  init(nodeDiv, data) {
    // Get our input elements.
    this.stepSize    = nodeDiv.querySelector('[id="' + this.id + '-stepSize"]');
    this.firstStep   = nodeDiv.querySelector('[id="' + this.id + '-firstStep"]');
    this.prevStep    = nodeDiv.querySelector('[id="' + this.id + '-prevStep"]');
    this.nextStep    = nodeDiv.querySelector('[id="' + this.id + '-nextStep"]');
    this.lastStep    = nodeDiv.querySelector('[id="' + this.id + '-lastStep"]');
    this.currentTime = nodeDiv.querySelector('[id="' + this.id + '-currentTime"]');
    this.syncSimTime = nodeDiv.querySelector('[id="' + this.id + '-syncSimulationTime"]');

    if (data.value) {
      this.syncSimTime.checked = data.syncSimTime;
      this.currentTime(data.currentTime);
      let step = data.intervals["intervals"][data.intervals["selectedIntervalIndex"]];
      this.setStepSize(step["size"], step["unit"]);
    }

    this.firstStep.addEventListener(
        "click", e => { CosmoScout.sendMessageToCPP({timeOperation: "first"}, this.parent.id); });

    this.prevStep.addEventListener(
        "click", e => { CosmoScout.sendMessageToCPP({timeOperation: "prev"}, this.parent.id); });

    this.nextStep.addEventListener(
        "click", e => { CosmoScout.sendMessageToCPP({timeOperation: "next"}, this.parent.id); });

    this.lastStep.addEventListener(
        "click", e => { CosmoScout.sendMessageToCPP({timeOperation: "last"}, this.parent.id); });

    this.syncSimTime.addEventListener("change", e => {
      CosmoScout.sendMessageToCPP({syncSimTime: this.syncSimTime.checked}, this.parent.id);
    });
  }

  /**
   * Displays the step size and converts the time unit to an easier
   * to understand one if it is in seconds.
   * @param {int} stepSize step size to display
   * @param {string} unit time unit of step size
   */
  setStepSize(stepSize, unit) {
    if (unit == "sec") {
      this.stepSize.innerHTML = this.secondsToTime(stepSize);
    } else {
      this.stepSize.innerHTML = stepSize + " " + unit;
    }
  }

  /**
   * Displays the time as the currently selected time of the node.
   * @param {string} time time to display
   */
  setCurrentTime(time) {
    this.currentTime.innerHTML = time;
  }

  /**
   * Converts seconds into an easier to understand time unit
   * @param {int} seconds The seconds to be converted
   * @returns string with time unit
   */
  secondsToTime(seconds) {
    const days             = Math.floor(seconds / (24 * 60 * 60));
    const hours            = Math.floor((seconds % (24 * 60 * 60)) / (60 * 60));
    const minutes          = Math.floor((seconds % (60 * 60)) / 60);
    const remainingSeconds = seconds % 60;

    let result = '';

    if (days > 0) {
      result += days + ' day' + (days > 1 ? 's' : '') + ' ';
    }

    if (hours > 0) {
      result += hours + ' hour' + (hours > 1 ? 's' : '') + ' ';
    }

    if (minutes > 0) {
      result += minutes + ' minute' + (minutes > 1 ? 's' : '') + ' ';
    }

    if (remainingSeconds > 0) {
      result += remainingSeconds + ' second' + (remainingSeconds > 1 ? 's' : '');
    }

    return result.trim();
  }
}
