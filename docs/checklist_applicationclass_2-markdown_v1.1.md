# Checklist for Application Class 2
> The checklist is in version 1.1. It refers to the DLR Software Engineering Guidelines [Reference Manual](https://doi.org/10.5281/zenodo.1344612) in version 1.0.0.

<!--
## Usage Hints
This checklist provides recommendations for software development. It is primarily intended for software developers for the self-assessment of developed software and as a source of ideas for further development. The checklist does not provide any new, revolutionary approaches to software development. However, it helps to not forget necessary, essential steps of software development. In addition, the recommendations can serve as an argumentation aid. 

The recommendations are created with a focus on know-how maintenance and good software engineering practice. They help to maintain the sustainability of the developed software. The recommendations encourage the use of tools, the creation of documentation, the establishment of processes and adherence to principles. When assessing a recommendation, it is recommended to consider to what extent the aspect mentioned is implemented and whether there is a need for improvement. This could be implemented as follows: 

* Is there currently no need for improvement and is the recommendation addressed appropriately? Status: **ok** 
* Is there any potential for improvement that should be implemented in the near future? Status: **todo**, record the need for improvement 
* Is the recommendation not yet relevant but could be helpful in a later development phase? Status: **future** 
* Is the recommendation not meaningfully implementable within the development context? Status: **n.a.** (not applicable) explain the reason 

In case of questions, you can contact the Software Engineering Contact of your institute or facility.

> Please note the status between "[]" and list remarks below a recommendation.

-->

## Summary of Results
The software CosmoScout VR implements 42 recommendations of application class 2. 

The focus of future improvements is on improving the testing capabilities and working out a requirements documentation.

## Table of Contents
[[Qualification](#qualifizierung)] [[Requirements Management](#anforderungsmanagement)] [[Software Architecture](#software-architektur)] [[Change Management](#aenderungsmanagement)] [[Design and Implementation](#design-implementierung)] [[Software Test](#software-test)] [[Release Management](#release-management)] [[Automation and Dependency Management](#automatisierung-abhaengigkeiten)] 

## Qualification <a name="qualifizierung"></a>
**EQA.1** - **[ok]** The software responsible recognises the different application classes and knows which is to be used for his/her software. *(from application class 1)*

**EQA.2** - **[ok]** The software responsible knows how to request specific support at the beginning and during development as well as to exchange ideas with other colleagues on the subject of software development. *(from application class 1)*

**EQA.3** - **[ok]** The persons involved in the development determine the skills needed with regard to their role and the intended application class. They communicate these needs to the supervisor. *(from application class 1)*

**EQA.4** - **[ok]** The persons involved in the development are given the tools needed for their tasks and are trained in their use. *(from application class 1)*

## Requirements Management <a name="anforderungsmanagement"></a>


**EAM.1** - **[todo]** The problem definition is coordinated with all parties involved and documented. It describes the objectives, the purpose of the software, the essential requirements and the desired application class in a concise, understandable way. *(from application class 1)*

**EAM.2** - **[todo]** Functional requirements are documented at least including a unique identifier, a description, the priority, the origin and the contact person. *(from application class 2)*

**EAM.3** - **[todo]** The constraints are documented. *(from application class 1)*

**EAM.4** - **[todo]** The quality requirements are documented and prioritised. *(from application class 2)*

**EAM.5** - **[todo]** User groups and their tasks are documented in the respective usage context. *(from application class 2)*

**EAM.8** - **[todo]** A glossary exists which describes the essential terms and definitions. *(from application class 2)*

**EAM.9** - **[todo]** The list of requirements is regularly coordinated, updated, analysed and checked. The resulting changes are traceable. *(from application class 2)*

## Software Architecture <a name="software-architektur"></a>
**ESA.1** - **[ok]** The architecture documentation is comprehensible for the relevant target groups. *(from application class 2)*

**ESA.2** - **[ok]** Essential architectural concepts and corresponding decisions are at least documented in a lean way. *(from application class 1)*

**ESA.3** - **[ok]** Testability of the software is appropriately addressed at software architecture level. *(from application class 2)*

**ESA.4** - **[ok]** The software architecture is coordinated with the relevant target groups. Changes are communicated actively and are comprehensible. *(from application class 2)*

**ESA.5** - **[ok]** The overlap between architectural documentation and implementation is minimised. *(from application class 2)*

**ESA.6** - **[todo]** The architecture documentation consistently uses the terminology of the requirements. *(from application class 2)*

**ESA.7** - **[todo]** Architectural concepts and decisions can be traced to requirements. *(from application class 2)*

**ESA.8** - **[ok]** Key architectural concepts are checked for their suitability using appropriate methods. *(from application class 2)*

**ESA.9** - **[ok]** The architecture documentation is updated regularly. *(from application class 2)*

## Change Management <a name="aenderungsmanagement"></a>
**EÄM.1** - **[ok]** The change process is coordinated in the development team and documented. *(from application class 2)*

**EÄM.2** - **[ok]** The most important information describing how to contribute to development are stored in a central location. *(from application class 1)*

**EÄM.3** - **[ok]** Change requests are centrally documented at least including a unique identifier, a short description and the contact details of the originator. They are stored long term and are searchable. In the case of bug reports, additional information about the reproducibility, the severity and the affected software version shall be documented. *(from application class 2)*

**EÄM.4** - **[ok]** A planning overview (roadmap) exists describing which software versions shall be achieved by when with which results. *(from application class 2)*

**EÄM.5** - **[ok]** Known bugs, important unresolved tasks and ideas are at least noted in bullet point form and stored centrally. *(from application class 1)*

**EÄM.6** - **[ok]** A detailed change history (change log) exists providing information about the functionalities and bug fixes of a software version. *(from application class 2)*

**EÄM.7** - **[ok]** A repository is set up in a version control system. The repository is adequately structured and ideally contains all artefacts for building a usable software version and for testing it. *(from application class 1)*

**EÄM.8** - **[ok]** Every change of the repository ideally serves a specific purpose, contains an understandable description and leaves the software in a consistent, working state. *(from application class 1)*

**EÄM.9** - **[ok]** If there are multiple common development branches, their purpose can be identified easily. *(from application class 2)*

## Design and Implementation <a name="design-implementierung"></a>
**EDI.1** - **[ok]** The usual patterns and solution approaches of the selected programming language are used and a set of rules regarding the programming style is consistently applied. The set of rules refers at least to the formatting and commenting. *(from application class 1)*

**EDI.2** - **[ok]** The software is structured modularly as far as possible. The modules are coupled loosely. I.e., a single module depends as little as possible on other modules. *(from application class 1)*

**EDI.3** - **[todo]** Ideally, there are module tests for every module. The module tests demonstrate their typical use and constraints. *(from application class 2)*

**EDI.4** - **[ok]** The implementation reflects the software architecture. *(from application class 2)*

**EDI.5** - **[ok]** It is continuously paid attention to room for improvement during development. Required changes (refactoring) may be implemented directly or prioritised through the change process. *(from application class 2)*

**EDI.6** - **[ok]** Suitability of rules with respect to the programming style is checked regularly. The preferred approaches and design patterns, relevant design principles and rules, as well as rules for permitted and non-permitted language elements may be supplemented. *(from application class 2)*

**EDI.7** - **[ok]** Adherence to simple rules concerning the programming style is checked or ensured automatically. *(from application class 2)*

**EDI.8** - **[todo]** Key design principles are defined and communicated. *(from application class 2)*

**EDI.9** - **[ok]** The source code and the comments contain as little duplicated information as possible. ("Don`t repeat yourself.") *(from application class 1)*

**EDI.10** - **[ok]** Prefer simple, understandable solutions. ("Keep it simple and stupid."). *(from application class 1)*

## Software Test <a name="software-test"></a>
**EST.1** - **[ok]** An overall test strategy is coordinated and defined. It is checked regularly for appropriateness. *(from application class 2)*

**EST.2** - **[ok]** Functional tests are systematically created and executed. *(from application class 2)*

**EST.4** - **[todo]** The basic functions and features of the software are tested in a near-operational environment. *(from application class 1)*

**EST.8** - **[ok]** The trend of the test results, the test coverage, the violations of the programming style, as well as the errors determined by code analysis tools is regularly examined for improvement. *(from application class 2)*

**EST.10** - **[ok]** The repository ideally contains all artefacts required to test the software. *(from application class 1)*

## Release Management <a name="release-management"></a>
**ERM.1** - **[ok]** Every release has a unique release number. The release number can be used to determine the underlying software version in the repository. *(from application class 1)*

**ERM.2** - **[ok]** The release package contains or references the user documentation. At least, it consists of installation, usage and contact information as well as release notes. In the case of the distribution of the release package to third parties outside DLR, the licensing conditions must be enclosed. *(from application class 1)*

**ERM.3** - **[ok]** Releases are published at regular, short intervals. *(from application class 2)*

**ERM.4** - **[ok]** The steps required for creating and approving a release are harmonised and documented. The release performance is largely automated. *(from application class 2)*

**ERM.6** - **[ok]** All foreseen test activities are executed during release performance. *(from application class 1)*

**ERM.7** - **[ok]** Prior to the approval of the release, all foreseen tests passed successfully. *(from application class 2)*

**ERM.9** - **[ok]** Prior to distribution of the release package to third parties outside DLR, it must be ensured that a licence is defined, that the licensing terms of used third-party software are met, and that all necessary licence information is included in the release package. *(from application class 1)*

**ERM.10** - **[ok]** Prior to distribution of the release package to third parties outside DLR, it has to be ensured that the export control regulations are met. *(from application class 1)*

## Automation and Dependency Management <a name="automatisierung-abhaengigkeiten"></a>
**EAA.1** - **[ok]** The simple build process is basically automated and necessary manual steps are described. In addition, there is sufficient information available about the operational and development environment. *(from application class 1)*

**EAA.2** - **[ok]** The dependencies to build the software are at least described by name, version number, purpose, licensing terms and reference source. *(from application class 1)*

**EAA.3** - **[ok]** New dependencies are checked for compatibility with the intended licence. *(from application class 2)*

**EAA.5** - **[ok]** In the build process, the execution of tests, the determination of metrics, the creation of the release package and, if necessary, other steps are performed automatically. *(from application class 2)*

**EAA.8** - **[ok]** An integration build is set up. *(from application class 2)*

**EAA.10** - **[ok]** The repository ideally contains all artefacts to perform the build process. *(from application class 1)*


> The checklist is in version 1.1. It bases on the document QMH-DLR-04-V03-Appendix in version 1.0.1.
