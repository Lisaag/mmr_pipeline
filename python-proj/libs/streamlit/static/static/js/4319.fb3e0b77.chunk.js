"use strict";(self.webpackChunk_streamlit_app=self.webpackChunk_streamlit_app||[]).push([[4319],{87814:function(e,t,i){i.d(t,{K:function(){return r}});var n=i(22951),l=i(91976),o=i(50641),r=function(){function e(){(0,n.Z)(this,e),this.formClearListener=void 0,this.lastWidgetMgr=void 0,this.lastFormId=void 0}return(0,l.Z)(e,[{key:"manageFormClearListener",value:function(e,t,i){null!=this.formClearListener&&this.lastWidgetMgr===e&&this.lastFormId===t||(this.disconnect(),(0,o.bM)(t)&&(this.formClearListener=e.addFormClearedListener(t,i),this.lastWidgetMgr=e,this.lastFormId=t))}},{key:"disconnect",value:function(){var e;null===(e=this.formClearListener)||void 0===e||e.disconnect(),this.formClearListener=void 0,this.lastWidgetMgr=void 0,this.lastFormId=void 0}}]),e}()},94319:function(e,t,i){i.r(t),i.d(t,{default:function(){return c}});var n=i(22951),l=i(91976),o=i(67591),r=i(94337),a=i(66845),s=i(25621),u=i(87814),d=i(97965),m=i(50641),p=i(40864),h=function(e){(0,o.Z)(i,e);var t=(0,r.Z)(i);function i(){var e;(0,n.Z)(this,i);for(var l=arguments.length,o=new Array(l),r=0;r<l;r++)o[r]=arguments[r];return(e=t.call.apply(t,[this].concat(o))).formClearHelper=new u.K,e.state={value:e.initialValue},e.commitWidgetValue=function(t){e.props.widgetMgr.setIntValue(e.props.element,e.state.value,t)},e.onFormCleared=function(){e.setState((function(e,t){var i;return{value:null!==(i=t.element.default)&&void 0!==i?i:null}}),(function(){return e.commitWidgetValue({fromUi:!0})}))},e.onChange=function(t){e.setState({value:t},(function(){return e.commitWidgetValue({fromUi:!0})}))},e}return(0,l.Z)(i,[{key:"initialValue",get:function(){var e,t=this.props.widgetMgr.getIntValue(this.props.element);return null!==(e=null!==t&&void 0!==t?t:this.props.element.default)&&void 0!==e?e:null}},{key:"componentDidMount",value:function(){this.props.element.setValue?this.updateFromProtobuf():this.commitWidgetValue({fromUi:!1})}},{key:"componentDidUpdate",value:function(){this.maybeUpdateFromProtobuf()}},{key:"componentWillUnmount",value:function(){this.formClearHelper.disconnect()}},{key:"maybeUpdateFromProtobuf",value:function(){this.props.element.setValue&&this.updateFromProtobuf()}},{key:"updateFromProtobuf",value:function(){var e=this,t=this.props.element.value;this.props.element.setValue=!1,this.setState({value:null!==t&&void 0!==t?t:null},(function(){e.commitWidgetValue({fromUi:!1})}))}},{key:"render",value:function(){var e=this.props.element,t=e.options,i=e.help,n=e.label,l=e.labelVisibility,o=e.formId,r=e.placeholder,a=this.props,s=a.disabled,u=a.widgetMgr,h=(0,m.le)(this.props.element.default)&&!s;return this.formClearHelper.manageFormClearListener(u,o,this.onFormCleared),(0,p.jsx)(d.ZP,{label:n,labelVisibility:(0,m.iF)(null===l||void 0===l?void 0:l.value),options:t,disabled:s,width:this.props.width,onChange:this.onChange,value:this.state.value,help:i,placeholder:r,clearable:h})}}]),i}(a.PureComponent),c=(0,s.b)(h)}}]);